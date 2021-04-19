import argparse, time, logging, os, sys
import copy
import math

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

import realnvp
# import utils
import os

if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=8,
    help='number of invertible blocks (default: 8)')
parser.add_argument(
    '--num-hidden',
    type=int,
    default=100)
parser.add_argument(
    '--seed', type=int, default=2018011309, help='random seed (default: 1)')

parser.add_argument(
    "--logdir", 
    default=None, 
    type=str)

parser.add_argument(
    "--gpuid",
    default=0,
    type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:"+str(args.gpuid) if args.cuda else "cpu")
CUDA = True if args.cuda else False

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.logdir is None:
    args.logdir = "Models-{}".format(time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(os.path.join(args.logdir, "models"), exist_ok=True)
os.makedirs(os.path.join(args.logdir, "images"), exist_ok=True)
print("Experiment dir : {}".format(args.logdir))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.logdir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# dataset = datasets.MNIST()
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
        torchvision.datasets.MNIST(root="../data", train=train, transform=mnist_transform),
        batch_size=batch_size,
        pin_memory=CUDA,
        drop_last=train,
        shuffle=train
    )

train_loader = load_mnist(train=True, batch_size=args.batch_size)
valid_loader = load_mnist(train=False, batch_size=args.batch_size)

num_inputs = 28 * 28
num_hidden = args.num_hidden
act = 'relu'

modules = []
masks = []

mask = torch.arange(0, num_inputs) % 2
mask = mask.to(device).float()
masks.extend([mask, 1 - mask])
mask2 = torch.zeros_like(mask)
mask2[: num_inputs//2] = 1
masks.extend([mask2, 1 - mask2])

for i in range(args.num_blocks):
    modules += [
        realnvp.CouplingLayer(
            num_inputs, num_hidden, masks[i % len(masks)]),
        realnvp.BatchNormFlow(num_inputs, 0.1),
        realnvp.Shuffle(num_inputs)
    ]

model = realnvp.FlowSequential(device, *modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        # nn.init.xavier_normal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

global_step = 0

def train(epoch):
    global global_step
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    idx = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        idx = batch_idx

        data = data.to(device)
        optimizer.zero_grad()
        loss = - model.log_probs(data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        pbar.set_postfix(lr=args.lr)
        
        global_step += 1
    
    logging.info('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (idx + 1)))

    pbar.close()


def validate(epoch, model, loader, prefix='Validation'):
    global global_step

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, (data, _) in enumerate(valid_loader):
        data = data.to(device)

        with torch.no_grad():
            val_loss += -model.log_probs(data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    pbar.close()
    ret = val_loss / len(loader.dataset)

    logging.info('Val, Log likelihood in nats: {:.6f}'.format(-ret))

    return ret


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    logging.info('\nEpoch: {}'.format(epoch))

    if epoch > 0 and epoch % 100 == 0:
        args.lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)
    if epoch % 10 == 0:
        save_path = os.path.join(args.logdir, "models/checkpoint{}.pt".format(epoch + 1))
        torch.save(model, save_path)
        logging.info("saving to {}".format(save_path))

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss

        save_path = os.path.join(args.logdir, "models/checkpoint_best.pt")
        torch.save(model, save_path)
        logging.info("best acc, saving to {}".format(save_path))

    logging.info(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))

    save_path = os.path.join(args.logdir, "images")
    model.save_images(epoch, save_path)

