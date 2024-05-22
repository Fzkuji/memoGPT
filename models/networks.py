import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


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


def log_memory_usage(tag):
    print(f"{tag} - Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
          f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


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
