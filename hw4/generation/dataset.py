import json
import os
import copy
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
import sys
sys.path.append(os.path.dirname(__file__))

cur_dir = os.path.dirname(os.path.abspath(__file__))
print(cur_dir)

class LMDataset(Dataset):

    def __init__(self, data_path=os.path.join(os.path.dirname(cur_dir), "Datasets/CCPC/"), vocab_file=os.path.join(os.path.dirname(cur_dir), "Datasets/CCPC/vocab.txt"), split="train", device="cpu"):

        self.filename = os.path.join(data_path, "ccpc_{}_v1.0.json".format(split))
        self.data = open(self.filename).readlines()
        self.vocab_file = vocab_file
        self.device = device
        self.split = split
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
    
    def __len__(self):
        return len(self.data)
    
    def _init_vocab(self):
        from tqdm import tqdm
        for line in tqdm(self.data, desc="initialize vocabulary"):
            poem = json.loads(line)
            all_words = poem['author'] + poem['content'] + poem['keywords'] + poem['title']
            for word in all_words:
                if word == "|" or word == " ":
                    continue
                self.dictionary.add_symbol(word)
        self.dictionary.save(self.vocab_file)

    # @profile
    def __getitem__(self, index):
        
        poem = json.loads(self.data[index])
        content = poem['content']
        content_id = self.dictionary.encode_line(content)
        return {
            "id": index,
            "length": len(content_id),
            "content": content_id
        }

    # @profile
    def collate_fn(self, samples):
        lens = [sample["length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)
        source = torch.LongTensor(bsz, max_len)
        source.fill_(self.dictionary.pad())
        source[:, 0].fill_(self.dictionary.bos())
        target = torch.ones_like(source) * self.dictionary.pad()

        ids, contents = [], []
        for idx, sample in enumerate(samples):
            ids.append(sample['id'])
            content = sample['content']
            source[idx, 1: sample["length"]] = content[: -1]
            target[idx, 0: sample["length"]] = content
        
        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "lengths": torch.LongTensor(lens).to(self.device),
            "source": source.to(self.device),
            "target": target.to(self.device)
        }

class Seq2SeqDataset(Dataset):

    def __init__(self, data_path=os.path.join(os.path.dirname(cur_dir), "Datasets/couplet/"), vocab_file=os.path.join(os.path.dirname(cur_dir), "Datasets/couplet/vocab.txt"), split="train", device="cpu"):

        self.split = split
        if self.split != "test":
            self.src_file = os.path.join(data_path, "train", "in.txt")
            self.tgt_file = os.path.join(data_path, "train", "out.txt")
        else:
            self.src_file = os.path.join(data_path, "test", "in.txt")
            self.tgt_file = os.path.join(data_path, "test", "out.txt")
            
        with open(self.src_file) as fsrc, open(self.tgt_file) as ftgt:
            self.src_lines = fsrc.readlines()
            self.tgt_lines = ftgt.readlines()
        if self.split == "train":
            self.src_lines = self.src_lines[: -5000]
            self.tgt_lines = self.tgt_lines[: -5000]
        elif self.split == "valid":
            self.src_lines = self.src_lines[-5000:]
            self.tgt_lines = self.tgt_lines[-5000:]

        assert len(self.src_lines) == len(self.tgt_lines)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
    
    def __len__(self):
        return len(self.src_lines)
    
    def _init_vocab(self):
        from tqdm import tqdm
        for src_line, tgt_line in tqdm(zip(self.src_lines, self.tgt_lines), desc="initialize vocabulary"):
            all_words = src_line.strip().split() + tgt_line.strip().split()

            for word in all_words:
                if word == "|" or word == " ":
                    continue
                self.dictionary.add_symbol(word)
        self.dictionary.save(self.vocab_file)

    # @profile
    def __getitem__(self, index):
        
        src_line = self.src_lines[index].strip().replace(" ", "")
        tgt_line = self.tgt_lines[index].strip().replace(" ", "")

        source_id = self.dictionary.encode_line(src_line)
        target_id = self.dictionary.encode_line(tgt_line)
        assert len(source_id) == len(target_id)
        return {
            "id": index,
            "length": len(source_id),
            "source": source_id,
            "target": target_id,
        }

    # @profile
    def collate_fn(self, samples):
        lens = [sample["length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)
        source = torch.LongTensor(bsz, max_len)

        source.fill_(self.dictionary.pad())
        source[:, 0].fill_(self.dictionary.bos())
        target = torch.ones_like(source) * self.dictionary.pad()
        prev_outputs = copy.deepcopy(source)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]

            source[idx, 1: sample["length"]] = source_ids[: -1]
            prev_outputs[idx, 1:sample["length"]] = target_ids[: -1]
            target[idx, 0: sample["length"]] = target_ids
        
        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "lengths": torch.LongTensor(lens).to(self.device),
            "source": source.to(self.device),
            "prev_outputs": prev_outputs.to(self.device),
            "target": target.to(self.device)
        }

if __name__ == "__main__":
    # dataset = LMDataset()
    dataset = Seq2SeqDataset(split="valid")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn)
    for sample in dataloader:
        print(sample)
    # for i in range(1000):
    #     print(dataset[i])
    print(len(dataset))
