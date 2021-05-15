
from typing import Dict, List, Optional, Tuple
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import random
import json

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

class LMModel(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.dictionary = dictionary
        
        self.config = Config(
            vocab_size=len(dictionary),
            max_position_embeddings=100,
            n_embed=args.embedding_dim,
            n_layer=args.num_layers,
            n_head=args.num_heads,
            pad_token_id=self.padding_idx,
            ffn_dim=args.hidden_size,
            )
        #embed_tokens = nn.Embedding(self.config.vocab_size, self.config.n_embed, self.padding_idx)
        self.endecoder = TransformerEncoderDecoder(self.config)
        self.out_proj = nn.Linear(args.embedding_dim, len(dictionary))

    def logits(self, source, **unused):
        real_source = torch.full_like(source, self.padding_idx)
        attention_mask = torch.full_like(source, True, dtype=torch.bool)
        hidden = self.endecoder(real_source, source, attention_mask=attention_mask)
        logits = self.out_proj(hidden)
        return logits
    
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


class Seq2SeqModel(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.dictionary = dictionary
        
        self.config = Config(
            vocab_size=len(dictionary),
            max_position_embeddings=100,
            n_embed=args.embedding_dim,
            n_layer=args.num_layers,
            n_head=8,
            pad_token_id=self.padding_idx,
            ffn_dim=args.hidden_size,
            )
        #embed_tokens = nn.Embedding(self.config.vocab_size, self.config.n_embed, self.padding_idx)
        self.endecoder = TransformerEncoderDecoder(self.config)
        self.out_proj = nn.Linear(args.embedding_dim, len(dictionary))

    def logits(self, source, prev_outputs, **unused):
        hidden = self.endecoder(source, prev_outputs)
        logits = self.out_proj(hidden)
        return logits
    
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


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

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

    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
    return decoder_input_ids, decoder_padding_mask, causal_mask

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.n_embed, padding_idx)

        self.encoder = TransformerEncoder(config, self.shared)
        self.decoder = TransformerDecoder(config, self.shared)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs: Optional[Tuple] = None,
        **kwargs,
    ):

        # make masks if user doesn't supply
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            causal_mask_dtype=self.shared.weight.dtype,
        )

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
        )

        return decoder_outputs


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


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embed

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.n_head,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.n_head,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        causal_mask=None,
        decoder_padding_mask=None,
    ):
        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
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
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.n_embed) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.n_embed, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.n_embed,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_layer)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.n_embed) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.n_embed) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_padding_mask: for ignoring pad tokens
            decoder_causal_mask: mask the future tokens
        """

        # embed positions
        positions = self.embed_positions(input_ids)

        x = self.embed_tokens(input_ids) * self.embed_scale

        x += positions
        x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            x = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                causal_mask=decoder_causal_mask,
            )

        if self.layer_norm:  
            x = self.layer_norm(x)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

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
