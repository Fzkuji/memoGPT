import math

from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, ROPE_INIT_FUNCTIONS
from transformers.utils import logging

from models.attentions.Memory import Memory
from models.utils import apply_rotary_emb, create_memory_mask, precompute_freqs_cis, apply_separate_rotary_emb

logger = logging.get_logger(__name__)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_separately(q, cos, sin, position_ids=None, unsqueeze_dim=1):

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MemorySelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.long_term_memory_update = False
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.head_dim * config.num_key_value_heads, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.head_dim * config.num_key_value_heads, bias=config.bias)

        # output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)  # 注意这边的bias，qwen2是False，其他模型可能会变
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.memory = Memory(config)
        self.register_buffer(
            "bias",
            create_memory_mask(
                sum(config.long_term_memory_size),
                config.short_term_memory_size,
                config.input_block_size,
                config.memory_block_size,
            )
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            dim=self.n_embd,
            max_position_embeddings=config.max_position_embeddings,
            base=10000,
            device=config.device,
            scaling_factor=1.0,
            rope_type="default",
            config=config,
        )

    def forward(self, x, short_term_memory_init=False, update_memory=False):

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # print("T: ", T)

        mid_pos = self.memory.max_len
        end_pos = self.memory.max_len + T

        if short_term_memory_init:

            if self.memory.short_term_memory.pool is None:

                self.memory.init_short_term_memory(x)

                q = self.q_proj(x).view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
                k = self.k_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                # print("q shape: ", q.shape)
                # print("k shape: ", k.shape)
                # print("v shape: ", v.shape)

                position_ids = torch.arange(self.config.short_term_memory_size, device=x.device).expand(B, -1)
                # print("position_ids shape: ", position_ids.shape)

                cos, sin = self.rotary_emb(v, position_ids)
                # print("cos shape: ", cos.shape)
                # print("sin shape: ", sin.shape)

                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

                # repeat k/v heads if n_kv_heads < n_heads
                k = repeat_kv(k, self.num_key_value_groups)
                v = repeat_kv(v, self.num_key_value_groups)

                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, mid_pos:mid_pos+self.config.short_term_memory_size, mid_pos:mid_pos+self.config.short_term_memory_size] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

                # output projection
                y = self.resid_dropout(self.o_proj(y))

                return y

            else:
                raise ValueError("Short term memory already initialized")

        else:

            short_term_memory = self.memory.get_short_term_memory(B)
            # print(self.memory.get_len())
            long_term_memory = self.memory.get_long_term_memory(B)
            memory_len = self.memory.get_len()

            # concatenate long_term_memory, short_term_memory and x
            if long_term_memory is not None:
                seq = torch.cat([long_term_memory, short_term_memory, x], dim=1)
            else:
                seq = torch.cat([short_term_memory, x], dim=1)

            q = self.q_proj(seq).view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
            k = self.k_proj(seq).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(seq).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # print("q shape: ", q.shape)
            # print("k shape: ", k.shape)
            # print("v shape: ", v.shape)

            position_ids = torch.arange(self.config.short_term_memory_size, self.config.short_term_memory_size + T, device=x.device).expand(B, -1)
            # print("position_ids shape: ", position_ids.shape)

            cos, sin = self.rotary_emb(v, position_ids)
            # print("cos shape: ", cos.shape)
            # print("sin shape: ", sin.shape)

            k[:, :, -T:, :] = apply_rotary_pos_emb_separately(k[:, :, -T:, :], cos, sin)

            position_ids = torch.arange(self.config.short_term_memory_size, self.config.short_term_memory_size + T, device=x.device).expand(B, -1,)
            cos, sin = self.rotary_emb(q, position_ids)
            q[:, :, -T:, :] = apply_rotary_pos_emb_separately(q[:, :, -T:, :], cos, sin)

            # repeat k/v heads if n_kv_heads < n_heads
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            # assert that q, k, v have the same shape else print the shape
            assert q.shape == k.shape == v.shape, f"q, k, v shapes are {q.shape}, {k.shape}, {v.shape}"

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # manual implementation of attention
            # print("memory_len: ", memory_len)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, mid_pos-memory_len:mid_pos+T, mid_pos-memory_len:mid_pos+T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

            if update_memory:
                # 断定短期记忆更新后和更新前不一样
                assert not torch.equal(short_term_memory, y[:, mid_pos - self.config.short_term_memory_size:mid_pos, :]), "Error: Short term memory is the same after update"

                self.memory.update_short_term_memory(y[:, -T - self.config.short_term_memory_size:-T, :])
                self.memory.update_long_term_memory(short_term_memory)

            # output projection
            y = y[:, -T:, :]  # only take the last T tokens
            y = self.resid_dropout(self.o_proj(y))

            return y





