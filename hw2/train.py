
import os
import argparse
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from PIL import Image
from cifar10_4x import CIFAR10_4x
from model import Net
from evaluation import evaluation

base_dir = os.path.dirname(__file__)

def set_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > (2**32 - 1):
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def main(args):
    bsz = args.batch_size
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.disable_cuda) else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])
    ])
    
    trainset = CIFAR10_4x(root=os.path.join(base_dir, 'data'), split="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=args.num_workers)

    validset = CIFAR10_4x(root=os.path.join(base_dir, 'data'), split='valid', transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=bsz, shuffle=False, num_workers=args.num_workers)

    net = Net()
    print("number of trained parameters: %d" % (sum([param.nelement() for param in net.parameters() if param.requires_grad])))
    print("number of total parameters: %d" % (sum([param.nelement() for param in net.parameters()])))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    net.to(device)
    best_acc = 0
    for epoch in range(args.num_epoch):  # loop over the dataset multiple times

        running_loss = deque([], maxlen=args.log_interval)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.append(loss.item())
            if i % args.log_interval == args.log_interval - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, sum(running_loss) / len(running_loss)))

        if epoch % args.save_interval == args.save_interval - 1:
            acc = evaluation(net, validloader, device)
            torch.save(net, os.path.join(args.model_dir, 'cifar10_4x_{}.pth'.format(epoch + 1)))
            if acc > best_acc:
                torch.save(net, os.path.join(args.model_dir, 'cifar10_4x_best.pth'))
                best_acc = acc
            net.train()

def get_args():
    curr_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,
                        default=os.path.join(curr_dir, "models"))
    parser.add_argument("--num-epoch", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--log-interval", default=100, type=int)
    parser.add_argument("--disable-cuda", default=False, action="store_true")
    parser.add_argument("--num-workers", default=1, type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)