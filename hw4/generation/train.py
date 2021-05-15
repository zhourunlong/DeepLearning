import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR
from torch.utils.data import DataLoader
from dataset import LMDataset, Seq2SeqDataset
from evaluation import evaluate
import argparse
import numpy as np
import os, sys, logging, time
from tqdm import tqdm

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
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=10, type=int)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--seq2seq", default=False, action="store_true")
    parser.add_argument("--model-type", default="transformer", choices=["lstm", "transformer"])
    parser.add_argument("--use-attention", default=False, action="store_true")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--cram", default=False, action="store_true")
    args = parser.parse_args()

    return args


def train(args):
    if args.logdir is None:
        args.logdir = "Models-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    task = "lm" if not args.seq2seq else "seq2seq"
    args.logdir += "_" + args.model_type + "_" + task
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "models"), exist_ok=True)
    print("Experiment dir : {}".format(args.logdir))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.logdir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    device = "cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu"

    mem_crammer = []

    if args.model_type == "lstm":
        from lstm import LMModel, Seq2SeqModel
    elif args.model_type == "transformer":
        from transformer import LMModel, Seq2SeqModel

    if args.seq2seq:
        train_set = Seq2SeqDataset(device=device)
        valid_set = Seq2SeqDataset(split="valid", device=device)
        model = Seq2SeqModel(args, train_set.dictionary).to(device)
    else:
        train_set = LMDataset(device=device)
        valid_set = LMDataset(split="valid", device=device)
        model = LMModel(args, train_set.dictionary).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    warmup_epoch = args.num_epoch * 0.1
    scheduler = ExponentialLR(optimizer, 0.1 ** (1 / (args.num_epoch - warmup_epoch)))
    iter_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)

    bestppl = 1e9
    for epoch in range(args.num_epoch):
        model.train()

        if args.cram:
            while True:
                try:
                    junk = torch.rand((9999, 9999), dtype=float, device=device)
                except:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    break
                mem_crammer.append(junk)

        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            for samples in pbar:
                if epoch < warmup_epoch:
                    warmup_scheduler.step()
                optimizer.zero_grad()

                while True:
                    success = True
                    try:
                        loss = model.get_loss(**samples)
                        loss.backward()
                        optimizer.step()
                    except:
                        del mem_crammer[-1]
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                        success = False
                        optimizer.zero_grad()
                    if success:
                        break

                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))


            logging.info("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))

        if epoch % args.save_interval == 0:
            savepath = os.path.join(args.logdir, "models/{}_{}.pt".format(args.model_type, epoch + 1))
            torch.save(model, savepath)
            logging.info("Saving to {}".format(savepath))
        
        if task == "lm":
            print("好    -->", model.generate("好", beam_size=3, device=device))
            print("秋水  -->", model.generate("秋水", beam_size=3, device=device))
            print("寒烟翠-->", model.generate("寒烟翠", beam_size=3, device=device))
        elif task == "seq2seq":
            print("改革春风吹满地-->", model.generate("改革春风吹满地", beam_size=2, device=device))
            print("牛津大学聪明人不及蟾蜍一半-->", model.generate("牛津大学聪明人不及蟾蜍一半", beam_size=2, device=device))
            print("一支穿云箭，青天白日重新现-->", model.generate("一支穿云箭，青天白日重新现", beam_size=2, device=device))
        
        loss, ppl = evaluate(model, valid_set, False)
        logging.info("Valid, Loss: %0.8f, ppl: %0.8f" % (loss, ppl))

        if ppl < bestppl:
            bestppl = ppl
            savepath = os.path.join(args.logdir, "models/{}_{}.pt".format(args.model_type, task))
            torch.save(model, savepath)
            logging.info("Best ppl! Saving to {}".format(savepath))
        
        if epoch >= warmup_epoch:
            scheduler.step()

if __name__ == "__main__":
    args = get_args()
    train(args)
