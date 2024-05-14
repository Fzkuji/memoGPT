from typing import Tuple

import torch


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


