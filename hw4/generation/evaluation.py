import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataset import *
import argparse
import os

@torch.no_grad()
def evaluate(model, dataset, printloss = True):
    model.eval()
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    ppls = []
    losses = []
    for samples in dataloader:
        bsz = len(samples['lengths'])
        logits = model.logits(**samples)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        entropy = F.nll_loss(
            lprobs, 
            samples["target"].view(-1),
            ignore_index=dataset.padding_idx,
            reduction="none"
        ).view(bsz, -1)
        ppl = torch.exp( entropy.sum(dim=-1, keepdim=True) / (samples["target"] != dataset.padding_idx).sum(dim=-1, keepdim=True).float() )
        ppls.extend(ppl.tolist())
        losses.append(entropy.mean().item())
    if printloss:
        print("%s: loss: %.3f, ppl: %.3f" % (dataset.split, np.mean(losses), np.mean(ppls)))

    return np.mean(losses), np.mean(ppls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--gpuid", type=int, default=0)
    args = parser.parse_args()

    loadpath = "models" if args.load_path is None else args.load_path

    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")

    basedir = os.path.dirname(os.path.abspath(__file__))
    for task in ["lm", "seq2seq"]:
        Dataset = LMDataset if task == "lm" else Seq2SeqDataset
        try:
            dataset = Dataset(split='test', device=device)
        except FileNotFoundError:
            dataset = Dataset(split="valid", device=device)
        for model_type in ["lstm", "transformer"]:
            model_name = "{}_{}.pt".format(model_type, task)

            try:
                model = torch.load(os.path.join(loadpath, model_name), map_location=device)
            except FileNotFoundError as e:
                print(e)
                continue
            print(task, model_type)
            evaluate(model, dataset)

            if hasattr(model, "generate"):
                if task == "lm":
                    print("好    -->", model.generate("好", beam_size=5, device=device))
                    print("秋水  -->", model.generate("秋水", beam_size=5, device=device))
                    print("寒烟翠-->", model.generate("寒烟翠", beam_size=5, device=device))
                elif task == "seq2seq":
                    print("改革春风吹满地-->", model.generate("改革春风吹满地", beam_size=5, device=device))
                    print("一支穿云箭，青天白日重新现-->", model.generate("一支穿云箭，青天白日重新现", beam_size=5, device=device))
                    print("图画里，龙不吟，虎不啸，小小书童可笑可笑-->", model.generate("图画里，龙不吟，虎不啸，小小书童可笑可笑", beam_size=5, device=device))
            print("-" * 50)
