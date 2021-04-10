import torch
import torch.nn as nn
import numpy as np
from collections import deque


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
    def __init__(self, model: MlpBackbone, device, buffer_size, langevin_k, langevin_noise_std, langevin_lr,
                 replay_p, lr, l2_coef=0., proj_norm=None):
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
        # TODO: implement langevin dynamics over the model
        return x

    def init_negative(self, batch_size):
        '''
        :param batch_size:
        :return: initial negative samples, a tensor of shape (batch_size,) + self.model.input_shape
        '''
        # TODO: initialize negative samples here.
        #  Sample from random noise with some probability; sample from self.replay_buffer otherwise.
        return x

    def train_step(self, batch_x: torch.Tensor):
        self.x_pos = batch_x
        self.x_neg_init = x_neg_init = self.init_negative(batch_x.shape[0])
        self.x_neg = x_neg = self.langevin_dynamic(x_neg_init)
        # TODO: write training objective here.
        loss = 0  # This is just a placeholder.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Fill replay buffer
        self.replay_buffer.extend([x_neg[i] for i in range(x_neg.shape[0])])
        # TODO: you may return anything you like for debugging
        return

    def inpainting(self, corrupted: torch.Tensor, mask):
        '''
        :param corrupted: images after adding noise, shape (batch_size C, H, W)
        :param mask: a binary tensor with the same size as ``corrupted''.  ``1'' positions indicate corrupted pixels.
                     ``0'' positions indicate ground truth pixels, which should not be changed during inpainting.
        :return: recovered images with the same size as ``corrupted''.
        '''
        recovered = corrupted
        return recovered

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
