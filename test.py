import torch


def precompute_freqs_cis(ranges: dict, dim: int):
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


# Example usage
ranges = {(-50, 0): 100, (0, 100): 1000}
dim = 10  # Dimensionality of the embeddings

embeddings = precompute_freqs_cis(ranges, dim)
print(embeddings.shape)  # Output: torch.Size([150, 5])
