import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import CLSDataset
from evaluation import evaluate
from model import Net
import argparse
import numpy as np
import os, sys, logging, time
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR
from transformers import AutoTokenizer, BertModel

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * (0.1 + 0.9 * self.last_epoch / (self.total_iters + 1e-8)) for base_lr in self.base_lrs]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=512, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=50, type=int)
    parser.add_argument("--save-interval", default=10, type=int)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--choice", type=int, default=4)
    parser.add_argument("--grad-acc-steps", type=int, default=1)
    parser.add_argument("--model-type", default="transformer", choices=["lstm", "transformer", "bert"])
    args = parser.parse_args()

    return args


def train(args):
    if args.logdir is None:
        args.logdir = "Models-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    args.logdir += "_" + args.model_type
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "models"), exist_ok=True)
    print("Experiment dir : {}".format(args.logdir))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.logdir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.gpuid >= 0:
        device = "cuda:" + str(args.gpuid)
    else:
        device = "cpu"
    
    '''
    train_set = CLSDataset(device=device, model_type=args.model_type)
    valid_set = CLSDataset(split="dev", device=device, model_type=args.model_type)
    
    torch.save(train_set, "train_set.pth")
    torch.save(valid_set, "valid_set.pth")
    '''

    train_set = torch.load("train_set.pth")
    train_set.settle(device, args.model_type)

    valid_set = torch.load("valid_set.pth")
    valid_set.settle(device, args.model_type)
    
    #train_set.summary()
    #valid_set.summary()
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
    
    model = Net(args, train_set.dictionary, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epoch = args.num_epoch * 0.1
    scheduler = ExponentialLR(optimizer, 0.1 ** (1 / (args.num_epoch - warmup_epoch)))
    iter_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)

    bestacc = 0
    step = 1
    for epoch in range(args.num_epoch):
        model.train()

        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            for step, samples in enumerate(pbar):
                if epoch < warmup_epoch:
                    warmup_scheduler.step()

                loss = model.get_loss(**samples)
                loss.backward()

                if step % args.grad_acc_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                
                step += 1
                
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))
            
            logging.info("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))

        if epoch % args.save_interval == 0:
            savepath = os.path.join(args.logdir, "models/{}.pt".format(epoch + 1))
            torch.save(model, savepath)
            logging.info("Saving to {}".format(savepath))

        loss, acc = evaluate(model, valid_set, False)
        logging.info("Valid, Loss: %0.8f, acc: %0.8f" % (loss, acc))

        if acc > bestacc:
            bestacc = acc
            savepath = os.path.join(args.logdir, "models/cls_best.pt")
            torch.save(model, savepath)
            logging.info("Best acc! Saving to {}".format(savepath))
        
        if epoch >= warmup_epoch:
            scheduler.step()

if __name__ == "__main__":
    args = get_args()
    train(args)
