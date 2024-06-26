import math

import torch
import torch.nn.functional as F
from torch import nn

from models.attentions.Memory import Memory
from models.utils import apply_rotary_emb, create_memory_mask, precompute_freqs_cis, apply_separate_rotary_emb


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

    def forward(self, x, short_term_memory_update=False):

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        end_pos = self.memory.max_len + T

        short_term_memory = self.memory.short_term_memory.get_all(B)

        # # concatenate the memory and the input
        # x = torch.cat([short_term_memory, x], dim=1)
        q_list, k_list, v_list = [], [], []
        for i in [short_term_memory, x]:
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q = self.q_proj(i)
            k = self.k_proj(i)
            v = self.v_proj(i)

            q_list.append(q.view(B, -1, self.num_attention_heads, self.head_dim))  # (B, T, nh, hs)
            k_list.append(k.view(B, -1, self.num_key_value_heads, self.head_dim))  # (B, T, nh, hs)
            v_list.append(v.view(B, -1, self.num_key_value_heads, self.head_dim))  # (B, T, nh, hs)

        long_k, long_v = self.memory.get_long_term_memory(B)
        start_pos = self.memory.get_long_term_memory_len()
        long_k.extend(k_list)
        long_v.extend(v_list)
        q = torch.cat(q_list, dim=1)
        k = torch.cat(long_k, dim=1)
        v = torch.cat(long_v, dim=1)

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





