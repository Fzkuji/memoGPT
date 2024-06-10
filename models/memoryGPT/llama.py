import math
import torch
import torch.nn.functional as F
from torch import nn
from models.attentions.Memory import Memory, MemoryConfig
from transformers.models.llama.modeling_llama import *

from models.utils import precompute_freqs_cis


class LlamaMemoryAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.memory_config = MemoryConfig(
            max_batch_size=config.max_batch_size,
            short_term_memory_size=config.short_term_memory_size,
            long_term_memory_layer=config.long_term_memory_layer,
            long_term_memory_chunk_size=config.long_term_memory_chunk_size,
            rope_theta=config.rope_theta,
            n_embd=config.n_embd,
            device=config.device,
        )
        self.memory = Memory(config.memory_config)
        self.short_term_memory_updated = False

        # 实例化RotaryEmbedding
        self.freqs_cis_memory = precompute_freqs_cis(
            dim=self.config.n_embd // self.config.n_head,
            fix_t=config.input_block_size,
            end=config.block_size * 2,
            theta=self.config.rope_theta,
        ).to(config.device)
        # print("MemorySelfAttention freqs_cis_memory shape: ", config.block_size)

        # 实例化RotaryEmbedding
        self.freqs_cis_seq = precompute_freqs_cis(
            dim=config.n_embd // config.n_head,
            end=config.block_size * 2,
            theta=config.rope_theta,
        ).to(config.device)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # get the memory
        short_query_states, short_key_states, short_value_states = self.memory.short_term_memory.get_all(bsz)
        long_query_states, long_key_states, long_value_states = self.memory.long_term_memory.get_all(bsz)

        if self.short_term_memory_updated:
            self.memory.update_long_term_memory(
                short_query_states,
                short_key_states,
                short_value_states,
                self.freqs_cis_memory,
            )
            self.short_term_memory_updated = False

        # concatenate the memory and the input
        query_states = torch.cat([long_query_states, short_query_states, query_states], dim=2)
        key_states = torch.cat([long_key_states, short_key_states, key_states], dim=2)
        value_states = torch.cat([long_value_states, short_value_states, value_states], dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # expand the front of the mask according to the length of memory (long + short)
            memory_length = long_key_states.shape[2] + short_key_states.shape[2]
            fill_value = attention_mask[0, 0, 0, 0].item()
            expanded_mask = torch.cat(
                [
                    torch.full((bsz, 1, 1, memory_length), fill_value=fill_value, device=attention_mask.device,
                               dtype=attention_mask.dtype),
                    causal_mask
                ],
                dim=-3,
            )
            attn_weights = attn_weights + expanded_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaWithMemory(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.layers):
            self.layers[i].attention = LlamaMemoryAttention(config)


# Example usage:
config = LlamaConfig.from_pretrained('llama-config-file')
model = LlamaWithMemory(config)