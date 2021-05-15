import json
import os
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

cur_dir = os.path.dirname(os.path.abspath(__file__))

class CLSDataset(Dataset):

    def __init__(self, data_path=os.path.join(os.path.dirname(cur_dir), "Datasets/CLS/"), vocab_file=os.path.join(os.path.dirname(cur_dir), "Datasets/CLS/vocab.txt"), split="train", device="cpu", model_type="transformer"):

        self.filename = os.path.join(data_path, "{}.json".format(split))

        with open(self.filename, encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.dictionary = Dictionary.load(vocab_file)
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.bert = (model_type == "bert")

        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

        self.cls_map = {
            "A": 0, "B": 1, "C": 2, "D": 3
        }
        self.pairs = []
        self.pairs_enc = []
        self.pairs_enc_bert = []
        with tqdm(self.data, desc=split) as pbar:
            for _, article in enumerate(pbar):
                content = article["Content"]
                content_enc = torch.cat((torch.IntTensor([self.dictionary.bos_index]), self.dictionary.encode_line(content)), dim=-1)
                content_enc_bert = torch.IntTensor(bert_tokenizer.encode(content))
                
                for question in article['Questions']:
                    q = question['Question']
                    q_enc = torch.cat((torch.IntTensor([self.dictionary.bos_index]), self.dictionary.encode_line(q)), dim=-1)
                    q_enc_bert = torch.IntTensor(bert_tokenizer.encode(q))

                    choices = question['Choices']
                    nchoices = []
                    nchoices_enc = []
                    nchoices_enc_bert = []

                    for choice in choices:
                        for ch in ["A", "B", "C", "D"]:
                            for space in [" ", ""]:
                                for punc in [".", "．", "、"]:
                                    choice = choice.strip().replace(ch + space + punc, "")
                        choice = choice.strip()

                        nchoices.append(choice)
                        nchoices_enc.append(self.dictionary.encode_line(choice))
                        nchoices_enc_bert.append(torch.IntTensor(bert_tokenizer.encode(choice)))

                    label = self.cls_map[question['Answer']]
                    self.pairs.append([content, q, nchoices, label])
                    self.pairs_enc.append([content_enc, q_enc, nchoices_enc, label])
                    self.pairs_enc_bert.append([content_enc_bert, q_enc_bert, nchoices_enc_bert, label])
                
                pbar.set_description("Initiating {} data".format(split))
        self.device = device
        self.split = split

    def settle(self, device, model_type):
        self.device = device
        self.bert = (model_type == "bert")
    
    def summary(self):
        max_l = 0
        s = {}
        for instance in self.pairs_enc:
            l = instance[0].shape[0]
            max_l = max(max_l, l)
            if l not in s:
                s[l] = 1
            else:
                s[l] += 1
        
        print(max_l)
        
        sum = 0
        for i in range(max_l, -1, -1):
            if i in s:
                sum += s[i]
            if i % 500 == 0:
                print(i, sum)

    def __len__(self):
        return len(self.pairs)

    # @profile
    def __getitem__(self, index):
        num = len(self.pairs[index][2])
        perm = torch.randperm(num)
        nchoices_enc = []
        for i in range(num):
            if self.bert:
                nchoices_enc.append(self.pairs_enc_bert[index][2][perm[i]])
            else:
                nchoices_enc.append(torch.cat((torch.IntTensor([self.dictionary.bos_index]), self.pairs_enc[index][2][perm[i]]), dim=-1))
        
        for i in range(num, 4, 1):
            if self.bert:
                nchoices_enc.append(torch.IntTensor([0]))
            else:
                nchoices_enc.append(torch.IntTensor([self.padding_idx]))

        for i in range(num):
            if perm[i] == self.pairs[index][3]:
                return {
                    "id": index,
                    "passage": self.pairs_enc_bert[index][0] if self.bert else self.pairs_enc[index][0],
                    "question": self.pairs_enc_bert[index][1] if self.bert else self.pairs_enc[index][1],
                    "choices": nchoices_enc,
                    "target": i
                }
        

    # @profile
    def collate_fn(self, samples):
        bsz = len(samples)
        ids = torch.zeros((bsz,), dtype=int, device=self.device)
        targets = torch.zeros((bsz,), dtype=int, device=self.device)
        for idx, sample in enumerate(samples):
            ids[idx] = sample["id"]
            targets[idx] = sample["target"]

        ret = {"id": ids, "targets": targets}

        for item in ["passage", "question", "choices"]:
            max_len = 0
            if item != "choices":
                for sample in samples:
                    max_len = max(max_len, sample[item].shape[0])
                source = torch.zeros((bsz, max_len), dtype=torch.long, device=self.device)
                if not self.bert:
                    source.fill_(self.dictionary.pad())
                for idx, sample in enumerate(samples):
                    source[idx, : sample[item].shape[0]] = sample[item]
            else:
                for i in range(4):
                    for sample in samples:
                        max_len = max(max_len, sample[item][i].shape[0])
                source = []
                for i in range(4):
                    source.append(torch.zeros((bsz, max_len), dtype=torch.long, device=self.device))
                    if not self.bert:
                        source[i].fill_(self.dictionary.pad())
                    for idx, sample in enumerate(samples):
                        source[i][idx, : sample[item][i].shape[0]] = sample[item][i]
            ret[item] = source
        
        return ret
