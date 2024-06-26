import torch


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


# Example usage
long_term_memory_size = 2
short_term_memory_size = 2
input_block_size = 4
memory_block_size = 4

mask = create_memory_mask(long_term_memory_size, short_term_memory_size, input_block_size, memory_block_size)
print(mask)
