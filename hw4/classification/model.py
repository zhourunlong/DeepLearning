from transformers import AutoTokenizer, BertModel
from typing import Dict, List, Optional, Tuple
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import random

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

def takeKey(x):
    return x[0]

class Config(object):
    dropout = 0.1
    attention_dropout = 0.0
    encoder_layerdrop = 0.0
    decoder_layerdrop = 0.0
    scale_embedding = None
    static_position_embeddings = False
    extra_pos_embeddings = 0
    normalize_before = False
    activation_function = "gelu"
    activation_dropout = 0.0
    normalize_embedding = True
    add_final_layer_norm = False
    init_std = 0.02

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class HiddenLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedder = nn.Embedding(config.vocab_size, config.n_embed, config.pad_token_id)
        self.encoder = nn.LSTM(input_size=config.n_embed, hidden_size=config.ffn_dim // 2, num_layers=config.n_layer, batch_first=True, bidirectional=True)
        self.hidden_size = config.ffn_dim // 2 * 2
        self.padding_idx = config.pad_token_id
    
    def forward(self, inputs):
        return self.encoder(self.embedder(inputs))[0]

class HiddenTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedder = nn.Embedding(config.vocab_size, config.n_embed, config.pad_token_id)
        self.encoder = TransformerEncoder(config, self.embedder)
        self.hidden_size = config.ffn_dim
        self.padding_idx = config.pad_token_id
    
    def forward(self, inputs):
        attention_mask = (inputs == self.padding_idx)
        return self.encoder(inputs, attention_mask)

class HiddenBert(nn.Module):

    def __init__(self, choice):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-chinese", output_hidden_states=True)
        self.hidden_size = 768
        self.choice = choice
    
    def forward(self, inputs):
        attention_mask = (inputs != 0)

        if inputs.shape[1] > 500:
            outputs = torch.zeros([inputs.shape[0], 500, 768], device=inputs.device)

            begpos = []
            for i in range(0, inputs.shape[1], 250):
                begpos.append(i)
            
            k = min(len(begpos), self.choice)
            p = np.random.choice(begpos, size=k, replace=False)
            
            for i in p:
                temporal = self.bert(inputs[:, i : i + 500], attention_mask[:, i : i + 500])[2][12]

                outputs[:, : temporal.shape[1], :] += temporal
            
            return outputs
        
        return self.bert(inputs, attention_mask)[2][12]

class Net(nn.Module):

    def __init__(self, args, dictionary, device):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.dictionary = dictionary
        self.device = device
        
        config = Config(
            vocab_size=len(dictionary),
            max_position_embeddings=2000,
            n_embed=args.embedding_dim,
            n_layer=args.num_layers,
            n_head=args.num_heads,
            pad_token_id=self.padding_idx,
            ffn_dim=args.hidden_size,
        )
        
        self.model_type = args.model_type

        if args.model_type == "lstm":
            self.encoder = HiddenLSTM(config)
        elif args.model_type == "transformer":
            self.encoder = HiddenTransformer(config)
        else:
            self.encoder = HiddenBert(args.choice)

        self.hidden_size = self.encoder.hidden_size
        hidden_size = self.hidden_size

        self.W, self.W1, self.W2, self.W3, self.W4 = {}, {}, {}, {}, {}
        for (x, y) in [("p", "q"), ("p", "a"), ("q", "a")]:
            self.W[x + y] = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
            self.W1[x + y] = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
            self.W2[x + y] = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
            self.W3[x + y] = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
            self.W4[x + y] = nn.Linear(hidden_size, hidden_size).to(device)
        self.V = nn.Linear(3 * hidden_size, 1)
    
    def set_device(self, device):
        self.device = device
        for (x, y) in [("p", "q"), ("p", "a"), ("q", "a")]:
            self.W[x + y].to(device)
            self.W1[x + y].to(device)
            self.W2[x + y].to(device)
            self.W3[x + y].to(device)
            self.W4[x + y].to(device)
        
        self.V.to(device)

    def logits(self, passage, question, choices, **unused):
        H, M = {}, {}
        H["p"], H["q"] = self.encoder(passage), self.encoder(question)
        C = torch.zeros([4, passage.shape[0], 3 * self.hidden_size], dtype=torch.float32, device=self.device)

        (x, y) = ("p", "q")
        score = torch.bmm(H[x], self.W[x + y](H[y]).transpose(1, 2))
        Gxy, Gyx = nn.Softmax(dim=-1)(score), nn.Softmax(dim=-2)(score)
        Ex, Ey = torch.bmm(Gxy, H[y]), torch.bmm(Gyx.transpose(1, 2), H[x])
        Sx, Sy = nn.ReLU()(self.W1[x + y](Ex)), nn.ReLU()(self.W2[x + y](Ey))
        Mx, My = torch.max(Sx, 1)[0], torch.max(Sy, 1)[0]
        g = nn.Sigmoid()(self.W3[x + y](Mx) + self.W4[x + y](My))
        M[x + y] = g * Mx + (1 - g) * My
        
        for i in range(4):
            H["a"] = self.encoder(choices[i])
            for (x, y) in [("p", "a"), ("q", "a")]:
                score = torch.bmm(H[x], self.W[x + y](H[y]).transpose(1, 2))
                Gxy, Gyx = nn.Softmax(dim=-1)(score), nn.Softmax(dim=-2)(score)
                Ex, Ey = torch.bmm(Gxy, H[y]), torch.bmm(Gyx.transpose(1, 2), H[x])
                Sx, Sy = nn.ReLU()(self.W1[x + y](Ex)), nn.ReLU()(self.W2[x + y](Ey))
                Mx, My = torch.max(Sx, 1)[0], torch.max(Sy, 1)[0]
                g = nn.Sigmoid()(self.W3[x + y](Mx) + self.W4[x + y](My))
                M[x + y] = g * Mx + (1 - g) * My
            C[i, :, :] = torch.cat((M["pq"], M["pa"], M["qa"]), dim=-1)

        logits = self.V(C.transpose_(0, 1)).squeeze(-1)
        return nn.Softmax(dim=-1)(logits)
    
    def get_loss(self, passage, question, choices, targets, **unused):
        logits = self.logits(passage, question, choices)
        loss = F.cross_entropy(logits, targets)
        return loss

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

def _prepare_decoder_inputs(
    config, input_ids, decoder_input_ids, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id

    bsz, tgt_len = decoder_input_ids.size()

    decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)

    # never mask leading token, even if it is pad
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]

    tmp = torch.full((tgt_len, tgt_len), -1e9, dtype=causal_mask_dtype, device=input_ids.device)
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
    return decoder_input_ids, decoder_padding_mask, causal_mask

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embed
        self.self_attn = Attention(self.embed_dim, config.n_head, dropout=config.attention_dropout)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: 
    """

    def __init__(self, config, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()

        self.layer_norm = LayerNorm(config.n_embed) if config.add_final_layer_norm else None

    def forward(
        self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        """
        # check attention mask and invert

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for encoder_layer in self.layers:

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                continue
            else:
                x = encoder_layer(x, attention_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        # calc k, v, q
        if self.encoder_decoder_attention:
            k, v = self.k_proj(key), self.v_proj(key)
        else:
            k, v = self.k_proj(query), self.v_proj(query)
        q = self.q_proj(query) * self.scaling

        # rearrange into (batch * #heads) * len * head_dim
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q = q.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # calc score
        score = torch.bmm(q, torch.transpose(k, 1, 2))
        # masked attention
        if attn_mask is not None:
            score += attn_mask.unsqueeze(0)
        # mask out padding
        if key_padding_mask is not None:
            score = score.view(bsz, self.num_heads, tgt_len, src_len)
            assert bsz == key_padding_mask.shape[0] and src_len == key_padding_mask.shape[1]
            score.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
            score = score.view(bsz * self.num_heads, tgt_len, src_len)
        score = torch.nn.Softmax(dim=2)(score)

        # dropout
        score = F.dropout(score, p=self.dropout, training=self.training)

        # calc attention
        att_hidden = torch.bmm(score, v)
        att_hidden = att_hidden.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        return self.out_proj(att_hidden)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]

        positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):

    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0 : dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
        out[:, dim // 2 :] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]

        # starts at 0, ends at 1-seq_len
        positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)
