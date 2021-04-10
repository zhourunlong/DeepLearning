import argparse
import copy
import math
import sys

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

import os

bsz = 100

def rescale(x, lo, hi):
    """Rescale a tensor to [lo,hi]."""
    assert(lo < hi), "[rescale] lo={0} must be smaller than hi={1}".format(lo,hi)
    old_width = torch.max(x)-torch.min(x)
    old_center = torch.min(x) + (old_width / 2.)
    new_width = float(hi-lo)
    new_center = lo + (new_width / 2.)
    # shift everything back to zero:
    x = x - old_center
    # rescale to correct width:
    x = x * (new_width / old_width)
    # shift everything to the new center:
    x = x + new_center
    # return:
    return x

def load_mnist(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess MNIST dataset."""
    mnist_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
        # rescale to [0.0001, 0.9999]:
        torchvision.transforms.Lambda(lambda x: rescale(x, 0.0001, 0.9999))
    ])
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root="./datasets", train=train, transform=mnist_transform),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=train
    )

train_loader = load_mnist(train=True, batch_size=bsz)

# model = torch.load("models4/realnvp/checkpoint101.pt")
model = torch.load("models/checkpoint_best.pt")

device = torch.device("cuda:0")
model.to(device)
model.eval()

dirname = "inpainting_images/"
os.makedirs(dirname, exist_ok=True)
for targets, cond_target in train_loader:
    mask = torch.zeros_like(targets).bool()
    num_samples, num_inputs = targets.size()
    inputs = copy.deepcopy(targets)

    B = torch.distributions.Bernoulli(torch.tensor([0.5]))
    # mask[:, torch.arange(num_inputs) % 56 < 28] = 1
    mask[:, num_inputs // 2 : ] = 1
    inputs[mask] = rescale( B.sample(inputs[mask].size()),  0.0001, 0.9999).squeeze(dim=-1)
    break

fixed_noise = model.prior.sample([num_samples, 28 * 28]).squeeze(dim=-1)
imgs = model.sample(bsz, noise=fixed_noise).detach().cpu()
imgs = torch.sigmoid(imgs.view(num_samples, 1, 28, 28)) 
torchvision.utils.save_image(imgs, os.path.join(dirname, 'img_sample.png'), nrow=int(np.sqrt(num_samples)))

torchvision.utils.save_image(targets.view(num_samples, 1, 28, 28), os.path.join(dirname, 'img_raw.png'), nrow=int(np.sqrt(num_samples)))
torchvision.utils.save_image(inputs.view(num_samples, 1, 28, 28), os.path.join(dirname, 'img_masked.png'), nrow=int(np.sqrt(num_samples)))

model.inpainting(num_samples, inputs, mask, dirname)
