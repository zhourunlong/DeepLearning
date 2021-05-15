import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataset import CLSDataset
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, BertModel

basedir = os.path.dirname(os.path.abspath(__file__))

@torch.no_grad()
def evaluate(model, dataset, printloss = True):
    model.eval()
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    losses = []
    hit = 0
    for samples in tqdm(dataloader, desc="validation"):
        targets = samples.pop("targets")
        
        logits = model.logits(**samples)
        predict_label = logits.argmax(dim=-1)
        hit += (predict_label == targets).sum()
        losses.append(F.cross_entropy(logits, targets).item())

    if printloss:
        print("%s: loss: %.3f, acc: %.3f" % (dataset.split, np.mean(losses), hit.item() / len(dataset)))

    return np.mean(losses), hit.item() / len(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, default=None)
    parser.add_argument("--gpuid", type=int, default=0)
    args = parser.parse_args()

    loadpath = basedir if args.load_dir is None else args.load_dir

    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")
    
    try:
        model = torch.load(os.path.join(loadpath, "models/cls_best.pt"), map_location=device)
        model.set_device(device)
    except FileNotFoundError as e:
        print(e)
        exit()

    try:
        dataset = CLSDataset(split='test', device=device, model_type=model.model_type)
    except FileNotFoundError:
        dataset = CLSDataset(split="dev", device=device, model_type=model.model_type)

    evaluate(model, dataset)
