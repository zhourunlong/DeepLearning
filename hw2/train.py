import os
import sys
import time
import argparse
import logging
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
from model_densenet import *
from model_resnet import *
from model_vgg import *
from model_vgg_ensemble import *
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

class PepperNoise(object):
    def __init__(self, prob = 0.1, density = 0.01):
        self.prob = prob
        self.density = density

    def __call__(self, rawimg):
        if np.random.random() > self.prob:
            return rawimg
        img = np.array(rawimg)
        rnd = np.random.rand(img.shape[0], img.shape[1])
        resultImg = img.copy()
        resultImg[rnd > 1 - self.density] = (255, 255, 255)
        ret = Image.fromarray(np.uint8(resultImg))
        return ret

def main(args):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    bsz = args.batch_size
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        #PepperNoise(prob = args.aug_prob),
        #transforms.Scale(160),
        transforms.RandomHorizontalFlip(args.aug_prob),
        #transforms.RandomVerticalFlip(args.aug_prob),
        #transforms.RandomRotation(20),
        #transforms.RandomCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])
    ])
    
    trainset = CIFAR10_4x(root=os.path.join(base_dir, 'data'), split="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=args.num_workers)

    transform_v = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])
    ])

    validset = CIFAR10_4x(root=os.path.join(base_dir, 'data'), split='valid', transform=transform_v)
    validloader = torch.utils.data.DataLoader(validset, batch_size=bsz, shuffle=False, num_workers=args.num_workers)

    best_acc = 0
    if args.cont_train:
        net = torch.load(args.model_dir + "/cifar10_4x_best.pth").to(device)
        best_acc = evaluation(net, validloader, device)
        net.train()
    else:
        net = VGG16_ensemble_small(args.ensemble, args.seed).to(device)
    print("number of trained parameters: %d" % (sum([param.nelement() for param in net.parameters() if param.requires_grad])))
    print("number of total parameters: %d" % (sum([param.nelement() for param in net.parameters()])))

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor = 0.5, patience=5)

    for epoch in range(args.num_epoch):  # loop over the dataset multiple times

        running_loss = 0
        cnt = 0
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
            running_loss += loss.item()
            cnt += 1
            if i % args.log_interval == args.log_interval - 1:
                logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / cnt))
        
        acc, val_loss = evaluation(net, validloader, device)
        tr_acc, _ = evaluation(net, trainloader, device)
        logging.info('epoch %d, learning rate %f, average training loss %f, training accuracy %f, valid loss %f, valid accuracy %f %%' % (epoch + 1, optimizer.param_groups[0]['lr'], running_loss / cnt, tr_acc, val_loss, acc))
        torch.save(net, os.path.join(args.model_dir, 'cifar10_4x_{}.pth'.format(epoch % 2)))
        logging.info('saving to {}/cifar10_4x_{}.pth'.format(args.model_dir, epoch % 2))
        if acc > best_acc:
            torch.save(net, os.path.join(args.model_dir, 'cifar10_4x_best.pth'))
            logging.info('best accuracy, saving to {}/cifar10_4x_best.pth'.format(args.model_dir))
            best_acc = acc
        net.train()

        scheduler.step(acc)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cont-train",action="store_true")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--num-epoch", default=100, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--seed", default=2018011309, type=int)
    parser.add_argument("--log-interval", default=500, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--aug-prob", default=0.5, type=float)
    parser.add_argument("--ensemble", default=6, type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    if not args.cont_train:
        args.model_dir = "Models-{}".format(time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.model_dir, exist_ok=True)
    print("Experiment dir : {}".format(args.model_dir))
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)