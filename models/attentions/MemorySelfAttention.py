import math

import torch
import torch.nn.functional as F
from torch import nn

from models.attentions.Memory import Memory
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
        self.long_term_memory_update = False
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

        # 实例化RotaryEmbedding
        self.freqs_cis_seq = precompute_freqs_cis(
            dim=config.n_embd // config.num_attention_heads,
            end=config.input_block_size + config.memory_block_size + config.short_term_memory_size,
            theta=config.rope_theta,
        ).to(config.device)

    def forward(self, x, short_term_memory_init=False, short_term_memory_update=False):

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        end_pos = self.memory.max_len + T

        if short_term_memory_init:

            if self.memory.short_term_memory.pool is None:

                q = self.q_proj(x).view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
                k = self.k_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                kv_seq_len = k.shape[-2]

                cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
                position_ids = torch.arange(self.config.short_term_memory_size, device=x.device).unsqueeze(0)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                # repeat k/v heads if n_kv_heads < n_heads
                k = repeat_kv(k, self.num_key_value_groups)
                v = repeat_kv(v, self.num_key_value_groups)

                # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
                # manual implementation of attention
                start_pos = end_pos - q.shape[2]
                start_pos_2 = end_pos - k.shape[2]

                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, start_pos:end_pos, start_pos_2:end_pos] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

                self.memory.init_short_term_memory(y)

                # output projection
                y = self.resid_dropout(self.o_proj(y))

                return y

            else:
                raise ValueError("Short term memory already initialized")

        else:
            short_term_memory = self.memory.short_term_memory.get_all(B)

            # # concatenate the memory and the input
            # x = torch.cat([short_term_memory, x], dim=1)
            q_list, k_list, v_list = [], [], []
            for i in [short_term_memory, x]:
                # calculate query, key, values for all heads in batch and move head forward to be the batch dim

                # # 这里不能循环赋值q, k, v，因为这会导致q, k, v的梯度无法传播
                # q = self.q_proj(i)
                # k = self.k_proj(i)
                # v = self.v_proj(i)

                q_list.append(self.q_proj(i).view(B, -1, self.num_attention_heads, self.head_dim))  # (B, T, nh, hs)
                k_list.append(self.k_proj(i).view(B, -1, self.num_key_value_heads, self.head_dim))  # (B, T, nh, hs)
                v_list.append(self.v_proj(i).view(B, -1, self.num_key_value_heads, self.head_dim))  # (B, T, nh, hs)

            long_k, long_v = self.memory.get_long_term_memory(B)
            start_pos = self.memory.get_long_term_memory_len()
            long_k.extend(k_list)
            long_v.extend(v_list)
            q = torch.cat(q_list, dim=1)
            k = torch.cat(long_k, dim=1)
            v = torch.cat(long_v, dim=1)

            # print("q.shape: ", q.shape)

            if self.long_term_memory_update:
                # 实例化RotaryEmbedding
                freqs_cis_memory = precompute_freqs_cis(
                    dim=self.config.n_embd // self.config.num_attention_heads,
                    fix_t=-T,
                    end=self.config.input_block_size * 2,
                    theta=self.config.rope_theta,
                ).to(self.config.device)

                self.memory.update_long_term_memory(
                    k[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # k
                    v[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # v
                    freqs_cis_memory,
                )
                self.long_term_memory_update = False
            # print("q.shape: ", q.shape)
            # print("freqs_cis_seq: ", self.freqs_cis_seq.shape)
            q = apply_separate_rotary_emb(q, freqs_cis=self.freqs_cis_seq[:q.shape[1]])

            k[:, -T - self.config.short_term_memory_size:, :, :] = apply_separate_rotary_emb(
                k[:, -T - self.config.short_term_memory_size:, :, :],
                freqs_cis=self.freqs_cis_seq[0: T + self.config.short_term_memory_size],
            )

            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, nh, T, hs)

            # repeat k/v heads if n_kv_heads < n_heads
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # manual implementation of attention
            start_pos = end_pos - q.shape[2]
            start_pos_2 = end_pos - k.shape[2]

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, start_pos:end_pos, start_pos_2:end_pos] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

            if short_term_memory_update:
                self.memory.update_short_term_memory(y[:, -T - self.config.short_term_memory_size:-T, :])
                self.long_term_memory_update = True

            # output projection
            y = y[:, -T:, :]  # only take the last T tokens
            y = self.resid_dropout(self.o_proj(y))

            return y





