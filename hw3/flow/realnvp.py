import math
import types
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions.normal import Normal
import torch.nn.init as init

class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct', **kwargs):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (- self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = torch.from_numpy(np.random.permutation(num_inputs))
        self.inv_perm = torch.argsort(self.perm)

    def forward(self, inputs, mode='direct'):
        bsz = inputs.shape[0]
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros((bsz, 1), device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros((bsz, 1), device=inputs.device)

class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, num_hidden, mask, s_act=nn.Tanh(), t_act=nn.ReLU()):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask
        
        n_hidden = 1

        s_net = [nn.Linear(num_inputs, num_hidden)]
        t_net = [nn.Linear(num_inputs, num_hidden)]
        for _ in range(n_hidden):
            s_net += [s_act, nn.Linear(num_hidden, num_hidden)]
            t_net += [t_act, nn.Linear(num_hidden, num_hidden)]
        s_net += [s_act, nn.Linear(num_hidden, num_inputs), s_act]
        t_net += [t_act, nn.Linear(num_hidden, num_inputs)]

        self.scale_net = nn.Sequential(*s_net)
        self.translate_net = nn.Sequential(*t_net)

    def forward(self, inputs, mode='direct'):
        mask = self.mask
        imask = 1 - mask

        masked_inputs = inputs * mask

        scaled = self.scale_net(masked_inputs)
        transed = self.translate_net(masked_inputs)

        log_det = (imask * scaled).sum(dim=1, keepdim=True)
        
        '''
        if mode == 'direct':
            outputs = masked_inputs + imask * (inputs * torch.exp(scaled) + transed)
            return outputs, log_det
        else:
            outputs = masked_inputs + imask * (inputs - transed) * torch.exp(-scaled)
            return outputs, -log_det
        '''

        if mode == 'direct':
            outputs = masked_inputs + imask * (inputs - transed) * torch.exp(-scaled)
            return outputs, -log_det
        else:
            outputs = masked_inputs + imask * (inputs * torch.exp(scaled) + transed)
            return outputs, log_det

class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """
    def __init__(self, device, *args):
        super().__init__(*args)
        self.prior = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

    def _pre_process(self, x):
        """

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): logits of `x`.

        See Also:
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        y = x.log() - (1. - x).log()
        torchvision.utils.save_image(y[: 100].sigmoid().view(100, 1, 28, 28), "test.png", nrow=10)

        return y

    def forward(self, inputs, mode='direct', logdets=None, **kwargs):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            if kwargs.get("pre_process", True):
                inputs = self._pre_process(inputs)
            for module in self._modules.values():
                inputs, logdet = module(inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, mode, **kwargs)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, pre_process=True):
        z, log_jacob = self(inputs, pre_process=pre_process)
        return self.prior.log_prob(z).sum(dim=1, keepdim=True) + log_jacob

    def sample(self, num_samples=None, noise=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward(noise, mode='inverse')[0]
        return samples

    #####################################################################################
    # If you want to edit this function, do not change the input and the output, otherwise, we might give you a zero in this task.
    def inpainting(self, num_samples, inputs, mask, savedir):
        from tqdm import tqdm
        os.makedirs(savedir, exist_ok=True)
        device = next(self.parameters()).device
        inputs = inputs.to(device).requires_grad_(True)
        inputs = inputs.log() - (1. - inputs).log()

        ep = torch.randn(inputs.size()).to(device)
        for i in tqdm(range(1000)):

            alpha =  0.2
            log_probs = self.log_probs(inputs, pre_process=False)
            dx = torch.autograd.grad([log_probs.sum()], [inputs])[0]
            dx = torch.clip(dx, -10, 10)

            with torch.no_grad():

                inputs[mask] += alpha * (dx[mask])
                inputs[mask] = torch.clip(inputs[mask], -10, 10)
            
            imgs = torch.sigmoid(inputs.view(num_samples, 1, 28, 28))
            if i % 10 == 0:
                torchvision.utils.save_image(imgs, os.path.join(savedir, 'img_{:03d}.png'.format(i + 1)), nrow=int(np.sqrt(num_samples)))
                alpha *= 0.99
        return imgs
    #############################################################################################

    def save_images(self, epoch, savedir):
        self.eval()
        # if epoch == 0:
        fixed_noise = self.prior.sample([100, 28 * 28]).squeeze()
        torchvision.utils.save_image(fixed_noise.view(100, 1, 28, 28), "noise.png", nrow=10)
        with torch.no_grad():

            imgs = self.sample(100, noise=fixed_noise).detach().cpu()

            imgs = torch.sigmoid(imgs.view(100, 1, 28, 28)) 
        
        os.makedirs(savedir, exist_ok=True)
        torchvision.utils.save_image(imgs, savedir + '/img_{:03d}.png'.format(epoch), nrow=10)
