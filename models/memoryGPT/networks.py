import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import functional as F
import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

from .memory import Memory
from .config import MemoryConfig
from .utils import precompute_freqs_cis, apply_rotary_emb


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def create_memory_mask(long_term_memory_size, short_term_memory_size, block_size):
    mask_size = long_term_memory_size + short_term_memory_size + block_size
    # Create a mask that is 1 in the lower left triangle and 0 in the upper right triangle
    mask = torch.tril(torch.ones(mask_size, mask_size))
    # Set the memory to 1
    mask[long_term_memory_size:long_term_memory_size + short_term_memory_size, :] = 1
    return mask.view(1, 1, mask_size, mask_size)


class MemorySelfAttention(nn.Module):

    def __init__(self, config: MemoryConfig):
        super().__init__()
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

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                create_memory_mask(
                    sum(config.long_term_memory_size),
                    config.short_term_memory_size,
                    config.block_size)
            )
            self.l_m_size = sum(config.long_term_memory_size)
            self.s_m_size = config.short_term_memory_size

        # 实例化RotaryEmbedding
        self.freqs_cis_memory = precompute_freqs_cis(
            dim=self.config.n_embd // self.config.n_head,
            fix_t=config.block_size,
            end=config.block_size * 2,
            theta=self.config.rope_theta,
        )

        # 实例化RotaryEmbedding
        self.freqs_cis_seq = precompute_freqs_cis(
            dim=config.n_embd // config.n_head,
            end=config.block_size * 2,
            theta=config.rope_theta,
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        short_term_memory = self.memory.short_term_memory.get_all(B).to(x.device)

        # concatenate the memory and the input
        x = torch.cat([short_term_memory, x], dim=1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, -1, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        q = q.view(B, -1, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        v = v.view(B, -1, self.n_head, C // self.n_head)  # (B, T, nh, hs)

        long_q, long_k, long_v = self.memory.get_long_term_memory(B)
        if long_q is not None:
            k = torch.cat([long_k.to(x.device), k], dim=1)
            v = torch.cat([long_v.to(x.device), v], dim=1)
            q = torch.cat([long_q.to(x.device), q], dim=1)
            start_pos = long_q.shape[1]
        else:
            start_pos = 0

        self.memory.update_long_term_memory(
            q[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # q
            k[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # k
            v[:, start_pos:start_pos+self.config.short_term_memory_size, :, :],  # v
            self.freqs_cis_memory.to(x.device),
        )

        apply_rotary_emb(
            q[:, -T - self.config.short_term_memory_size:, :, :],
            k[:, -T - self.config.short_term_memory_size:, :, :],
            freqs_cis=self.freqs_cis_seq[0: T + self.config.short_term_memory_size].to(x.device),
        )

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            return "Flash not implemented yet!"
            # # efficient attention using Flash Attention CUDA kernels
            # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
            #                                                      dropout_p=self.dropout if self.training else 0,
            #                                                      is_causal=True)
        else:
            # manual implementation of attention
            end_pos = self.l_m_size + self.s_m_size + T
            start_pos = end_pos - q.shape[2]
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, start_pos:end_pos, start_pos:end_pos] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

        self.memory.update_short_term_memory(y[:, -T - self.config.short_term_memory_size:-T, :])

        # output projection
        y = y[:, -T:, :]  # only take the last T tokens
        y = self.resid_dropout(self.c_proj(y))

        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=2, dtype=torch.float).to(inputs.dtype)

        # print("weights shape:", weights.shape)
        # print("weights:", weights)
        # print("selected_experts", selected_experts.shape)
        # print("selected_experts shape", selected_experts)

        results = torch.zeros_like(inputs)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_token, nth_expert = torch.where(selected_experts == i)
            expert_out = expert(inputs[batch_idx, nth_token])

            results[batch_idx, nth_token] += weights[batch_idx, nth_token, nth_expert].unsqueeze(-1) * expert_out

        return results
