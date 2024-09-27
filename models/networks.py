import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

from transformers.activations import ACT2FN


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# class MLP(nn.Module):
#
#     def __init__(self, config):
#         super().__init__()
#         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
#         self.gelu = nn.GELU()
#         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
#         self.dropout = nn.Dropout(config.dropout)
#
#     def forward(self, x):
#         x = self.c_fc(x)
#         x = self.gelu(x)
#         x = self.c_proj(x)
#         x = self.dropout(x)
#         return x


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
    def __init__(self, experts: List[nn.Module], gate_proj: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate_proj = gate_proj
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate_proj(inputs)
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


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))