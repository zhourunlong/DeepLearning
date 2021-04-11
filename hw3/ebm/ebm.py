import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class MlpBackbone(nn.Module):
    def __init__(self, input_shape, hidden_size, activation=nn.functional.leaky_relu):
        super(MlpBackbone, self).__init__()
        self.input_shape = input_shape  # (C, H, W)
        self.hidden_size = hidden_size
        # Layers
        self.fc1 = nn.Linear(np.prod(self.input_shape), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

        self.activation = activation

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        out = self.fc4(x)
        return out


class Trainer(object):
    def __init__(self, model: MlpBackbone, device, seed, buffer_size, langevin_k, langevin_noise_std, langevin_lr, replay_p, lr, l2_coef, proj_norm=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.model = model
        self.device = device
        self.replay_buffer = deque(maxlen=buffer_size)
        self.langevin_k = langevin_k
        self.langevin_noise_std = langevin_noise_std
        self.langevin_lr = langevin_lr
        self.replay_p = replay_p
        self.l2_coef = l2_coef
        self.proj_norm = proj_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0., 0.999))  # Follow the paper.
        self.x_pos = None  # Positive samples.
        self.x_neg_init = None  # Initial negative samples.
        self.x_neg = None  # Negative samples.

    def langevin_dynamic(self, x: torch.Tensor):
        '''
        :param x: initial negative sample
        :return: the resulting negative sample
        '''
        x.requires_grad_(True)
        for p in self.model.parameters():
            p.requires_grad = False

        for i in range(self.langevin_k):
            out = self.model(x)
            out.sum().backward()

            x.data.add_(x.grad.data, alpha = -self.langevin_lr)
            x.grad.zero_()

            noise = torch.randn(x.shape[0], 1, 28, 28, device=self.device)
            noise.normal_(0, self.langevin_noise_std)

            x.data.add_(noise)

            x.data.clamp_(0, 1)

        for p in self.model.parameters():
            p.requires_grad = True

        return x.detach()

    def init_negative(self, batch_size):
        '''
        :param batch_size:
        :return: initial negative samples, a tensor of shape (batch_size,) + self.model.input_shape
        '''
        if len(self.replay_buffer) < 1:
            return torch.rand(batch_size, 1, 28, 28, device=self.device)
        
        n_replay = (np.random.rand(batch_size) < self.replay_p).sum()

        replay_sample = random.sample(self.replay_buffer, n_replay)
        replay_sample = torch.stack(replay_sample)
        random_sample = torch.rand(batch_size - n_replay, 1, 28, 28, device=self.device)

        return torch.cat([replay_sample, random_sample], 0)

    def train_step(self, pos_img: torch.Tensor):
        neg_init = self.init_negative(pos_img.shape[0])
        neg_img = self.langevin_dynamic(neg_init)
        
        self.optimizer.zero_grad()

        pos_out = self.model(pos_img)
        neg_out = self.model(neg_img)

        loss = pos_out - neg_out + self.l2_coef * (pos_out ** 2 + neg_out ** 2)
        loss = loss.mean()
        loss.backward()

        self.optimizer.step()

        self.replay_buffer.extend(neg_img)
        return loss

    def inpainting(self, corrupted: torch.Tensor, mask):
        '''
        :param corrupted: images after adding noise, shape (batch_size C, H, W)
        :param mask: a binary tensor with the same size as ``corrupted''.  ``1'' positions indicate corrupted pixels.
                     ``0'' positions indicate ground truth pixels, which should not be changed during inpainting.
        :return: recovered images with the same size as ``corrupted''.
        '''

        bsz = corrupted.shape[0]

        x = []
        for j in range(bsz):
            x.append(corrupted[j].clone())
            x[j].requires_grad_(True)
        for p in self.model.parameters():
            p.requires_grad = False

        for i in range(self.langevin_k):
            for j in range(bsz):
                out = self.model(x[j])
                out.backward()
                x[j].data.add_(x[j].grad.data * mask[j], alpha = -self.langevin_lr)
                x[j].grad.zero_()
                x[j].data.clamp_(0, 1)

        for p in self.model.parameters():
            p.requires_grad = True

        return torch.stack(x).detach()

    def save(self, save_path):
        save_dict = {'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'replay_buffer': self.replay_buffer,
                     }
        torch.save(save_dict, save_path)

    def load(self, load_pth, evaluate=True):
        checkpoint = torch.load(load_pth, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if evaluate:
            self.model.eval()
        else:
            self.model.train()
