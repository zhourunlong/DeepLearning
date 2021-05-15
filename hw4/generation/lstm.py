import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq

def takeKey(x):
    return x[0]

class BaseModel(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.dictionary = dictionary

class LMModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.embedder = nn.Embedding(len(dictionary), args.embedding_dim, self.padding_idx)
        self.lstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_first=True)
        self.output_project = nn.Linear(args.hidden_size, len(dictionary))

    def logits(self, source, **unused):
        embed = self.embedder(source)
        out, (_, _) = self.lstm(embed)
        return self.output_project(out)
    
    def get_loss(self, source, target, reduce=True, **unused):
        logits = self.logits(source)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs, 
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, prefix, max_len=100, beam_size=None, device=None):
        '''
        prefix: The initial words, like "白"
        
        output a string like "白日依山尽，黄河入海流，欲穷千里目，更上一层楼。"
        '''
        if beam_size is None:
            beam_size = 5

        prefix_len = len(prefix)
        raw_source = self.dictionary.encode_line(prefix).long().to(device)
        source = torch.zeros_like(raw_source)
        for i in range(prefix_len):
            source[i + 1] = raw_source[i]
        source[0] = self.dictionary.bos()

        q = [(0, source.unsqueeze(0))]

        answer_pool = []
        for i in range(prefix_len, max_len, 1):
            p = []
            for (neg_log_p, prev_outputs) in q:
                logits = self.logits(prev_outputs)
                lprobs = F.log_softmax(logits, dim=-1).squeeze(0)
                val, idx = torch.topk(lprobs[i, :], beam_size)
                
                for j in range(beam_size):
                    if idx[j] == self.dictionary.eos():
                        answer_pool.append(((neg_log_p - val[j]) / (i - prefix_len + 1), prev_outputs))
                        continue
                    nxt = torch.cat((prev_outputs, idx[j].view(1, 1)), dim=-1)
                    p.append((neg_log_p - val[j], nxt))
            
            p.sort(key=takeKey)
            q = p[: beam_size]

            answer_pool.sort(key=takeKey)
            answer_pool = answer_pool[: beam_size]

        #answer_pool += q
        #answer_pool.sort(key=takeKey)

        output = ""
        (neg_log_p, embed) = answer_pool[0]
        for i in range(1, embed.shape[1], 1):
            output += self.dictionary[embed[0, i]]

        return output


class Seq2SeqModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.use_attention = args.use_attention
        self.embedder = nn.Embedding(len(dictionary), args.embedding_dim, self.padding_idx)
        self.encoder = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_first=True)
        if self.use_attention:
            self.score_weight = nn.Linear(args.hidden_size, args.hidden_size)
            self.output_project = nn.Linear(2 * args.hidden_size, len(dictionary))
        else:
            self.output_project = nn.Linear(args.hidden_size, len(dictionary))

    def logits(self, source, prev_outputs, **unused):
        embed, embed_prev = self.embedder(source), self.embedder(prev_outputs)
        enc_hidden, (h, c) = self.encoder(embed)
        dec_hidden, (_, _) = self.decoder(embed_prev, (h, c))
        
        if self.use_attention:
            score = torch.bmm(dec_hidden, torch.transpose(self.score_weight(enc_hidden), 1, 2))
            score = torch.nn.Softmax(dim=2)(score)
            att_hidden = torch.bmm(score, enc_hidden)
            dec_hidden = torch.cat((att_hidden, dec_hidden), dim=2)

        return self.output_project(dec_hidden)
    
    def get_loss(self, source, prev_outputs, target, reduce=True, **unused):
        logits = self.logits(source, prev_outputs)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs, 
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, inputs, max_len=100, beam_size=None, device=None):
        '''
        inputs, 上联: "改革春风吹满地"
        
        output, 下联: "复兴政策暖万家"
        '''
        if beam_size is None:
            beam_size = 5

        seqlen = len(inputs)
        raw_source = self.dictionary.encode_line(inputs).long().to(device)
        source = torch.zeros_like(raw_source)
        for i in range(seqlen):
            source[i + 1] = raw_source[i]
        source[0] = self.dictionary.bos()
        source.unsqueeze_(0)
        
        init = torch.full(source.shape, self.padding_idx, dtype=torch.long, device=device)
        init[0, 0] = self.dictionary.bos()
        q = [(0, init)]

        for i in range(seqlen):
            p = []
            for (neg_log_p, prev_outputs) in q:
                logits = self.logits(source, prev_outputs)
                lprobs = F.log_softmax(logits, dim=-1).squeeze(0)
                val, idx = torch.topk(lprobs[i, :], beam_size)
                
                for j in range(beam_size):
                    if idx[j] < 5:
                        continue
                    nxt = prev_outputs.clone()
                    nxt[0, i + 1] = idx[j]
                    p.append((neg_log_p - val[j], nxt))
            
            p.sort(key=takeKey)
            q = p[: beam_size]

        output = ""
        (neg_log_p, embed) = q[0]
        for i in range(seqlen):
            output += self.dictionary[embed[0, i + 1]]

        return output
