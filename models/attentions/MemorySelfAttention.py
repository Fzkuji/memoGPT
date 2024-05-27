import math

import torch
import torch.nn.functional as F
from torch import nn

from models.attentions.Memory import Memory
from models.utils import apply_rotary_emb, create_memory_mask, precompute_freqs_cis


class MemorySelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.short_term_memory_updated = False
        self.config = config
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.memory = Memory(config)
        self.register_buffer(
            "bias",
            create_memory_mask(
                sum(config.long_term_memory_size),
                config.short_term_memory_size,
                config.block_size)
        )

        # 实例化RotaryEmbedding
        self.freqs_cis_memory = precompute_freqs_cis(
            dim=self.config.n_embd // self.config.n_head,
            fix_t=config.block_size,
            end=config.block_size * 2,
            theta=self.config.rope_theta,
        ).to(config.device)

        # 实例化RotaryEmbedding
        self.freqs_cis_seq = precompute_freqs_cis(
            dim=config.n_embd // config.n_head,
            end=config.block_size * 2,
            theta=config.rope_theta,
        ).to(config.device)

    def forward(self, x):

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        end_pos = self.memory.max_len + T

        short_term_memory = self.memory.short_term_memory.get_all(B)

        # # concatenate the memory and the input
        # x = torch.cat([short_term_memory, x], dim=1)
        q_list, k_list, v_list = [], [], []
        for i in [short_term_memory, x]:
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v = self.c_attn(i).split(self.n_embd, dim=2)

            k_list.append(k.view(B, -1, self.n_head, C // self.n_head))  # (B, T, nh, hs)
            q_list.append(q.view(B, -1, self.n_head, C // self.n_head))  # (B, T, nh, hs)
            v_list.append(v.view(B, -1, self.n_head, C // self.n_head))  # (B, T, nh, hs)

        long_q, long_k, long_v = self.memory.get_long_term_memory(B)
        start_pos = self.memory.get_long_term_memory_len()
        long_q.extend(q_list)
        long_k.extend(k_list)
        long_v.extend(v_list)
        k = torch.cat(long_k, dim=1)
        v = torch.cat(long_v, dim=1)
        q = torch.cat(long_q, dim=1)

        if self.short_term_memory_updated:
            self.memory.update_long_term_memory(
                q[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # q
                k[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # k
                v[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # v
                self.freqs_cis_memory,
            )
            # print("Updated long term memory")

        apply_rotary_emb(
            q[:, -T - self.config.short_term_memory_size:, :, :],
            k[:, -T - self.config.short_term_memory_size:, :, :],
            freqs_cis=self.freqs_cis_seq[0: T + self.config.short_term_memory_size].to(x.device),
        )

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        start_pos = end_pos - q.shape[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, start_pos:end_pos, start_pos:end_pos] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

        if T == self.config.block_size:
            self.memory.update_short_term_memory(y[:, -T - self.config.short_term_memory_size:-T, :])
            self.short_term_memory_updated = True
            # print("Updated short term memory")

        # output projection
        y = y[:, -T:, :]  # only take the last T tokens
        y = self.resid_dropout(self.c_proj(y))

        return y


