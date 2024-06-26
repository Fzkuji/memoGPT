from typing import Tuple

import math
import torch


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr=1e-5):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device, dtype=torch.float32)
#     freqs = torch.outer(t, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis


def precompute_freqs_cis(dim: int, fix_t: int = None, start: int = 0, end: int = 4096, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if fix_t is not None:
        t = torch.full((end - start,), fix_t, device=freqs.device, dtype=torch.float32)
    else:
        t = torch.arange(start, end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_separate_rotary_emb(
        xq: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> torch.Tensor:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def apply_rotary_emb_inplace(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor
) -> None:
    """
    Apply rotary embeddings in-place on the input tensors.

    Args:
        xq (torch.Tensor): Input tensor for queries of shape (..., length, n_embd).
        xk (torch.Tensor): Input tensor for keys of shape (..., length, n_embd).
        freqs_cis (torch.Tensor): Precomputed rotary embeddings.
    """
    # Ensure the input tensors are of float type for complex operations
    xq_float = xq.float()
    xk_float = xk.float()

    # Convert input tensors to complex
    xq_complex = torch.view_as_complex(xq_float.reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk_float.reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis to match the shape of the input tensors
    freqs_cis_broadcasted = reshape_for_broadcast(freqs_cis, xq_complex)

    # Apply rotary embedding in-place
    xq_rotated = torch.view_as_real(xq_complex * freqs_cis_broadcasted).flatten(3)
    xk_rotated = torch.view_as_real(xk_complex * freqs_cis_broadcasted).flatten(3)

    # Copy the result back to the original input tensors to avoid extra memory usage
    xq.copy_(xq_rotated.type_as(xq))
    xk.copy_(xk_rotated.type_as(xk))


def precompute_memory_freqs_cis(ranges: dict, dim: int):
    """
    Precompute the frequency tensors for the positional encoding

    # Example usage
    ranges = {(-50, 0): 100, (0, 100): 1000}
    dim = 10  # Dimensionality of the embeddings

    embeddings = precompute_freqs_cis(ranges, dim)
    print(embeddings.shape)  # Output: torch.Size([150, 5])
    """
    freqs_list = []
    for range_, theta in ranges:
        start, end = range_
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(start, end, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        freqs_list.append(freqs_cis)

    # Concatenate the frequency tensors along the time dimension
    freqs_concatenated = torch.cat(freqs_list, dim=0)
    return freqs_concatenated


def create_memory_mask(long_term_memory_size, short_term_memory_size, input_block_size, memory_block_size):
    mask_size = long_term_memory_size + short_term_memory_size + input_block_size + memory_block_size
    # Create a mask that is 1 in the lower left triangle and 0 in the upper right triangle
    mask = torch.tril(torch.ones(mask_size, mask_size))
    # Set the memory to 1
    mask[long_term_memory_size:long_term_memory_size + short_term_memory_size, :] = 1

    # Modify the lower left corner of the combined input and memory block part to set the lower left memory_block_size x memory_block_size triangle to 0
    start_idx = long_term_memory_size + short_term_memory_size
    end_idx = start_idx + input_block_size + memory_block_size
    for i in range(memory_block_size):
        for j in range(memory_block_size - i):
            mask[end_idx - i - 1, start_idx + j] = 0

    return mask.view(1, 1, mask_size, mask_size)


# def create_memory_mask(short_term_memory_size, input_block_size, memory_block_size, max_position_embeddings=32768):
#     input_height = input_block_size + memory_block_size
#     mask_height = short_term_memory_size + input_block_size + memory_block_size
#     width = max_position_embeddings * 2  # max_position_embeddings = 32768
#     # Create a mask that is 1 in the lower left triangle and 0 in the upper right triangle
#     memory_mask = torch.ones(short_term_memory_size, width)
#     input_mask = torch.tril(torch.ones(input_height, width), diagonal=1)
#     # Set the memory to 1
#     mask = torch.cat([memory_mask, input_mask], dim=0)
#     return mask.view(1, 1, mask_height, -1)
