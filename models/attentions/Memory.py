import torch
import torch.nn as nn

from models.memoryGPT.config import GPTConfig
from models.utils import apply_rotary_emb


# class MemoryPool(nn.Module):
#     """ A simple pool for storing tensors with a fixed capacity """
#
#     def __init__(self, config, capacity, attention_head_dims, key_value_head_dims, max_batch_size=64):
#         super(MemoryPool, self).__init__()
#         self.batch_size = max_batch_size
#         self.capacity = capacity
#
#         # initialize the pool with zeros
#         self.pool_q = torch.zeros(max_batch_size, capacity, attention_head_dims)
#         self.pool_k = torch.zeros(max_batch_size, capacity, key_value_head_dims)
#         self.pool_v = torch.zeros(max_batch_size, capacity, key_value_head_dims)
#
#     def get_all(self, batch_size):
#         """ Return a tensor containing all elements in the pool """
#         assert batch_size <= self.batch_size, f"Batch size {batch_size} is greater than the maximum batch size {self.batch_size}"
#         # return self.pool[:batch_size]
#         return self.pool_q[:batch_size], self.pool_k[:batch_size], self.pool_v[:batch_size]
#
#     def get_len(self):
#         return self.capacity
#
#     def update(self, tensor_q, tensor_k, tensor_v):
#         """ Update the pool with a new tensor """
#         # assert tensor_q.shape == tensor_k.shape == tensor_v.shape, "All tensors should have the same shape"
#
#         bsz, seqlen, *dim = tensor_q.shape
#         assert bsz <= self.batch_size, f"Batch size {bsz} is greater than the maximum batch size {self.batch_size}"
#         assert seqlen == self.capacity, f"Sequence length {seqlen} is not equal to the capacity {self.capacity}"
#
#         self.pool_q[:bsz, :, :] = tensor_q.detach()
#         self.pool_k[:bsz, :, :] = tensor_k.detach()
#         self.pool_v[:bsz, :, :] = tensor_v.detach()
#
#     def clear(self):
#         """ Clear the pool """
#         self.pool_q.fill_(0)  # Efficient way to reset all elements to zero
#         self.pool_k.fill_(0)
#         self.pool_v.fill_(0)


class MemoryPool(nn.Module):
    """ A simple pool for storing tensors with a fixed capacity """

    def __init__(self, config, capacity, *tensor_dims, max_batch_size=64):
        super(MemoryPool, self).__init__()
        self.batch_size = max_batch_size
        self.capacity = capacity
        self.tensor_dims = tensor_dims
        # initialize the pool with zeros
        self.pool = torch.zeros(max_batch_size, capacity, *tensor_dims)
        self.pool = self.pool.to(config.device)
        # self.pool_k = torch.zeros(max_batch_size, capacity, *tensor_dims)
        # self.pool_v = torch.zeros(max_batch_size, capacity, *tensor_dims)

    def get_all(self, batch_size):
        """ Return a tensor containing all elements in the pool """
        assert batch_size <= self.batch_size, f"Batch size {batch_size} is greater than the maximum batch size {self.batch_size}"
        return self.pool[:batch_size]
        # return self.pool_q[:batch_size], self.pool_k[:batch_size], self.pool_v[:batch_size]

    def get_len(self):
        return self.capacity

    # def update(self, tensor_q, tensor_k, tensor_v):
    def update(self, tensor):
        """ Update the pool with a new tensor """
        # assert tensor_q.shape == tensor_k.shape == tensor_v.shape, "All tensors should have the same shape"

        bsz, seqlen, *dim = tensor.shape
        assert bsz <= self.batch_size, f"Batch size {bsz} is greater than the maximum batch size {self.batch_size}"
        assert seqlen == self.capacity, f"Sequence length {seqlen} is not equal to the capacity {self.capacity}"

        self.pool[:bsz, :, :] = tensor.detach()
        # self.pool_k[:bsz, :, :] = tensor_k
        # self.pool_v[:bsz, :, :] = tensor_v

    def clear(self):
        """ Clear the pool """
        self.pool.fill_(0)  # Efficient way to reset all elements to zero
        # self.pool_k.fill_(0)
        # self.pool_v.fill_(0)


class MemoryQueue(nn.Module):
    """ A simple queue for storing tensors with a fixed capacity

    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    queue = MemoryQueue(5, 3, 224, 224)  # For example, queue of 5 tensors with shape [3, 224, 224]

    # Simulating adding tensors to the queue
    for _ in range(10):  # Add more than the capacity to test the FIFO functionality
        new_tensor = torch.randn(3, 224, 224).to(device)  # Random tensor with expected dimensions
        queue.push(new_tensor)

    # Fetch all tensors in the queue as a single stacked tensor
    all_tensors = queue.get_all()
    print(all_tensors.shape)  # Should show (5, 3, 224, 224) indicating the queue is holding 5 tensors
    """

    def __init__(self, config, capacity, *tensor_dims, max_batch_size=64):
        super(MemoryQueue, self).__init__()
        self.batch_size = max_batch_size
        self.capacity = capacity
        self.tensor_dims = tensor_dims  # Dimensions of the tensor you expect to store
        # self.queue_q = []  # Max: torch.zeros(max_batch_size, capacity, *tensor_dims)
        self.queue_k = []  # Max: torch.zeros(max_batch_size, capacity, *tensor_dims)
        self.queue_v = []
        self.index = 0

    def push(self, tensor_k, tensor_v):
        """ Add a tensor to the queue """
        bsz, seqlen, *dim = tensor_k.shape  # bsz: batch size, seqlen: sequence length
        if bsz != self.batch_size:
            self.clear()

        self.queue_k.append(tensor_k.detach())
        self.queue_v.append(tensor_v.detach())

        if len(self.queue_k) > self.capacity:
            self.queue_k.pop(0)
            self.queue_v.pop(0)
            self.index += 1

        if self.index > self.capacity:
            self.index = self.index - self.capacity
            return True  # Carry over
        else:
            return False

    def update_rotary_emb(self, freqs_cis):
        if self.queue_k is None:
            return
        for i in range(len(self.queue_k)):
            k, v = apply_rotary_emb(self.queue_k[i], self.queue_v[i], freqs_cis[0:self.queue_k[i].shape[1]])
            self.queue_k[i], self.queue_v[i] = k, v

    def get_all(self, batch_size=0):
        """ Return a tensor containing all elements in the queue """
        if self.queue_k and batch_size != self.queue_k[0].shape[0]:
            self.clear()
        return self.queue_k, self.queue_v

    def get_len(self):
        return len(self.queue_k)

    def clear(self):
        """ Clear the queue """
        self.queue_k = []
        self.queue_v = []
        self.index = 0


class Memory(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Memory, self).__init__()
        self.config = config
        self.max_len = sum(config.long_term_memory_size) + config.short_term_memory_size

        # assert config.long_term_memory_size[0] % config.short_term_memory_size == 0, "Long-term memory size should be a multiple of short-term memory size"
        self.theta_step = config.long_term_memory_size[0] // config.short_term_memory_size

        # Initialize the long-term memory with MemoryQueue
        self.long_term_memory = nn.ModuleList(
            [
                MemoryQueue(self.config, i, config.n_embd, max_batch_size=config.max_batch_size)
                for i in config.long_term_memory_size
            ]
        )

        # Initialize the short-term memory with MemoryQueue
        self.short_term_memory = MemoryPool(self.config, config.short_term_memory_size, config.n_embd, max_batch_size=config.max_batch_size)

    def update_long_term_memory(self, tensor_k, tensor_v, freqs_cis):
        for memory in self.long_term_memory:
            memory.update_rotary_emb(freqs_cis)
        for memory in self.long_term_memory:
            if tensor_k is not None:
                carry_over = memory.push(tensor_k.detach(), tensor_v.detach())
                if not carry_over:
                    break
        torch.cuda.empty_cache()

    def update_short_term_memory(self, tensor):
        self.short_term_memory.update(tensor.detach())
        torch.cuda.empty_cache()

    def get_long_term_memory(self, batch_size):
        all_k = []
        all_v = []

        for memory in self.long_term_memory[::-1]:
            k, v = memory.get_all(batch_size)
            all_k.extend(k)
            all_v.extend(v)

        return all_k, all_v

    def get_long_term_memory_len(self):
        return sum([memory.get_len() for memory in self.long_term_memory]) * self.short_term_memory.capacity

    def get_len(self):
        short_term_len = self.short_term_memory.get_len()
        long_term_len = self.get_long_term_memory_len()
        return short_term_len + long_term_len

    def clear_all(self):
        for memory in self.long_term_memory:
            memory.clear()
        self.short_term_memory.clear()
