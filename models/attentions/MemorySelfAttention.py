import math

import torch
import torch.nn.functional as F
from torch import nn

from models.attentions.Memory import Memory
from models.memoryGPT.gpt2 import MLP, RMSNorm
from models.utils import apply_rotary_emb, create_memory_mask, precompute_freqs_cis, apply_separate_rotary_emb


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_separately(q, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding with Mixtral->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


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

        self.q_memo_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_memo_proj = nn.Linear(config.n_embd, self.head_dim * config.num_key_value_heads, bias=config.bias)
        self.v_memo_proj = nn.Linear(config.n_embd, self.head_dim * config.num_key_value_heads, bias=config.bias)

        # output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)  # 注意这边的bias，qwen2是False，其他模型可能会变

        self.o_memo_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

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

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            # max_position_embeddings=self.max_position_embeddings,
            max_position_embeddings=32768,
            base=config.rope_theta,
        )

        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)

    def init_memo_proj(self):
        # 令memo_proj的参数和q,k,v_proj的参数相同
        # 使用 copy_ 方法复制权重，而不是直接赋值
        self.q_memo_proj.weight.data.copy_(self.q_proj.weight.data)
        self.q_memo_proj.bias.data.copy_(self.q_proj.bias.data)
        self.k_memo_proj.weight.data.copy_(self.k_proj.weight.data)
        self.k_memo_proj.bias.data.copy_(self.k_proj.bias.data)
        self.v_memo_proj.weight.data.copy_(self.v_proj.weight.data)
        self.v_memo_proj.bias.data.copy_(self.v_proj.bias.data)
        self.o_memo_proj.weight.data.copy_(self.o_proj.weight.data)

    def forward(self, x, short_term_memory_init=False, update_memory=False, x_pre_norm=None):

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # print("T: ", T)

        mid_pos = self.memory.max_len
        end_pos = self.memory.max_len + T

        if short_term_memory_init:

            if self.memory.short_term_memory.pool is None:

                self.memory.init_short_term_memory(x)

                q = self.q_memo_proj(x).view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
                k = self.k_memo_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                v = self.v_memo_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                kv_seq_len = k.shape[-2]

                cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
                position_ids = torch.arange(self.config.short_term_memory_size, device=x.device).unsqueeze(0)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

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
                y = self.resid_dropout(self.o_memo_proj(y))
                y = x_pre_norm + y

                # mlp
                y = self.post_attention_layernorm(y)
                y = self.mlp(y)

                return y

            else:
                raise ValueError("Short term memory already initialized")

        else:

            short_term_memory = self.memory.get_short_term_memory(B)
            long_term_memory = self.memory.get_long_term_memory(B)
            memory_len = self.memory.get_len()

            # calculate long_term_memory and short_term_memory together
            if long_term_memory is not None:
                seq = torch.cat([long_term_memory, short_term_memory], dim=1)
            else:
                seq = short_term_memory

            q_memory = self.q_memo_proj(seq).view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
            k_memory = self.k_memo_proj(seq).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v_memory = self.v_memo_proj(seq).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            q = self.q_proj(x).view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
            k = self.k_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            q = torch.cat([q_memory, q], dim=2)
            k = torch.cat([k_memory, k], dim=2)
            v = torch.cat([v_memory, v], dim=2)

            # print('kv_seq_len: ', k.shape[-2])
            # print('q_seq_len: ', q.shape[-2])

            kv_seq_len = k.shape[-2]
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            # position_ids 为 short_term_memory 到 short_term_memory + T 的位置
            position_ids = torch.arange(self.config.short_term_memory_size, self.config.short_term_memory_size + T, device=x.device).unsqueeze(0)
            k[:, :, -T:, :] = apply_rotary_pos_emb_separately(k[:, :, -T:, :], cos, sin, position_ids)

            q_seq_len = q.shape[-2]
            cos, sin = self.rotary_emb(q, seq_len=q_seq_len)
            # position_ids 为 short_term_memory 到 short_term_memory + T 的位置
            position_ids = torch.arange(self.config.short_term_memory_size, self.config.short_term_memory_size + T, device=x.device).unsqueeze(0)
            q[:, :, -T:, :] = apply_rotary_pos_emb_separately(q[:, :, -T:, :], cos, sin, position_ids)

            # repeat k/v heads if n_kv_heads < n_heads
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, mid_pos-memory_len:mid_pos+T, mid_pos-memory_len:mid_pos+T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

            if update_memory:
                """更新长期记忆"""
                self.memory.update_long_term_memory(short_term_memory)

                """更新短期记忆"""
                # o_memo_proj
                short_term_memory = self.resid_dropout(self.o_memo_proj(y[:, -T - self.config.short_term_memory_size:-T, :]))

                # x + o_memo_proj
                short_term_memory = x_pre_norm + short_term_memory

                # mlp
                short_term_memory = self.post_attention_layernorm(short_term_memory)
                short_term_memory = x_pre_norm + self.mlp(short_term_memory)

                # input_layer_norm
                short_term_memory = self.input_layernorm(short_term_memory)

                self.memory.update_short_term_memory(short_term_memory)

            # output projection
            y = y[:, -T:, :]  # only take the last T tokens
            y = self.resid_dropout(self.o_proj(y))

            return y





