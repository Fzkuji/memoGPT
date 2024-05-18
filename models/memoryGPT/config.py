from dataclasses import dataclass


@dataclass
class MemoryConfig:
    max_batch_size: int = 64

    memory_dim: int = 384
    short_term_memory_size: int = 4
    long_term_memory_layer: int = 4
    long_term_memory_chunk_size: int = 8
    long_term_memory_size = ([short_term_memory_size * long_term_memory_chunk_size] * long_term_memory_layer +
                             [(short_term_memory_size - 1) * long_term_memory_chunk_size])

    rope_theta: float = 500000
    block_size: int = 4096
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    use_moe: bool = False
    n_expert: int = 16
    n_expert_per_tok: int = 4
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
