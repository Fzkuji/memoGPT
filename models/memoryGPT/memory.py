import torch
import torch.nn as nn

from .config import MemoryConfig
from .utils import apply_rotary_emb


class MemoryPool(nn.Module):
    """ A simple pool for storing tensors with a fixed capacity """

    def __init__(self, capacity, *tensor_dims, max_batch_size=64):
        super(MemoryPool, self).__init__()
        self.batch_size = max_batch_size
        self.capacity = capacity
        self.tensor_dims = tensor_dims
        # initialize the pool with zeros
        self.pool = torch.zeros(max_batch_size, capacity, *tensor_dims)
        # self.pool_k = torch.zeros(max_batch_size, capacity, *tensor_dims)
        # self.pool_v = torch.zeros(max_batch_size, capacity, *tensor_dims)

    def get_all(self, batch_size):
        """ Return a tensor containing all elements in the pool """
        assert batch_size <= self.batch_size, f"Batch size {batch_size} is greater than the maximum batch size {self.batch_size}"
        return self.pool[:batch_size]
        # return self.pool_q[:batch_size], self.pool_k[:batch_size], self.pool_v[:batch_size]

    # def update(self, tensor_q, tensor_k, tensor_v):
    def update(self, tensor):
        """ Update the pool with a new tensor """
        # assert tensor_q.shape == tensor_k.shape == tensor_v.shape, "All tensors should have the same shape"

        bsz, seqlen, *dim = tensor.shape
        assert bsz <= self.batch_size, f"Batch size {bsz} is greater than the maximum batch size {self.batch_size}"
        assert seqlen == self.capacity, f"Sequence length {seqlen} is not equal to the capacity {self.capacity}"

        self.pool[:bsz, :, :] = tensor
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

    def __init__(self, capacity, *tensor_dims, max_batch_size=64):
        super(MemoryQueue, self).__init__()
        self.batch_size = max_batch_size
        self.capacity = capacity
        self.tensor_dims = tensor_dims  # Dimensions of the tensor you expect to store
        self.queue_q = None  # Max: torch.zeros(max_batch_size, capacity, *tensor_dims)
        self.queue_k = None  # Max: torch.zeros(max_batch_size, capacity, *tensor_dims)
        self.queue_v = None
        self.index = 0

    def push(self, tensor_q, tensor_k, tensor_v):
        """ Add a tensor to the queue """
        bsz, seqlen, *dim = tensor_k.shape  # bsz: batch size, seqlen: sequence length
        assert bsz <= self.batch_size, f"Batch size {bsz} is greater than the maximum batch size {self.batch_size}"

        # Pad the tensor if the batch size is less than the maximum batch size
        if bsz < self.batch_size:
            tensor_q = torch.cat([tensor_q, torch.zeros(self.batch_size - bsz, seqlen, *dim).to(tensor_q.device)], dim=0)
            tensor_k = torch.cat([tensor_k, torch.zeros(self.batch_size - bsz, seqlen, *dim).to(tensor_k.device)], dim=0)
            tensor_v = torch.cat([tensor_v, torch.zeros(self.batch_size - bsz, seqlen, *dim).to(tensor_v.device)], dim=0)

        if self.queue_q is None:
            self.queue_q = tensor_q
            self.queue_k = tensor_k
            self.queue_v = tensor_v
        else:
            self.queue_q = torch.cat([self.queue_q, tensor_q], dim=1)
            self.queue_k = torch.cat([self.queue_k, tensor_k], dim=1)
            self.queue_v = torch.cat([self.queue_v, tensor_v], dim=1)

        if self.queue_q.shape[1] > self.capacity:
            self.queue_q = self.queue_q[:, -self.capacity:, :]
            self.queue_k = self.queue_k[:, -self.capacity:, :]
            self.queue_v = self.queue_v[:, -self.capacity:, :]

        self.index = self.index + seqlen
        if self.index > self.capacity:
            self.index = self.index - self.capacity
            return True  # Carry over
        else:
            return False

    def update_rotary_emb(self, freqs_cis):
        if self.queue_q is None:
            return
        k, v = apply_rotary_emb(self.queue_k, self.queue_v, freqs_cis[0:self.queue_q.shape[1]])
        self.queue_k, self.queue_v = k, v

    def get_all(self, batch_size=0):
        """ Return a tensor containing all elements in the queue """
        if self.queue_q is None:
            return None, None, None
        else:
            return self.queue_q[:batch_size], self.queue_k[:batch_size], self.queue_v[:batch_size]

    def get_len(self):
        if self.queue_k is None:
            return 0
        return self.queue_k.shape[1]

    def clear(self):
        """ Clear the queue """
        self.queue_q = None
        self.queue_k = None
        self.queue_v = None
        self.index = 0


class Memory(nn.Module):
    def __init__(self, config: MemoryConfig):
        super(Memory, self).__init__()
        self.config = config

        # assert config.long_term_memory_size[0] % config.short_term_memory_size == 0, "Long-term memory size should be a multiple of short-term memory size"
        self.theta_step = config.long_term_memory_size[0] // config.short_term_memory_size

        # Initialize the long-term memory with MemoryQueue
        self.long_term_memory = nn.ModuleList(
            [
                MemoryQueue(i, config.memory_dim, max_batch_size=config.max_batch_size)
                for i in config.long_term_memory_size
            ]
        )

        # Initialize the short-term memory with MemoryQueue
        self.short_term_memory = MemoryPool(config.short_term_memory_size, config.memory_dim, max_batch_size=config.max_batch_size)

    def update_long_term_memory(self, tensor_q, tensor_k, tensor_v, freqs_cis):
        for memory in self.long_term_memory:
            memory.update_rotary_emb(freqs_cis)
        for memory in self.long_term_memory:
            if tensor_q is not None:
                carry_over = memory.push(tensor_q, tensor_k, tensor_v)
                if not carry_over:
                    break
        # detach the memory from the computation graph
        for memory in self.long_term_memory:
            if memory.queue_q is not None:
                memory.queue_q = memory.queue_q.detach()
                memory.queue_k = memory.queue_k.detach()
                memory.queue_v = memory.queue_v.detach()

    def update_short_term_memory(self, tensor):
        self.short_term_memory.update(tensor)
        # detach the memory from the computation graph
        self.short_term_memory.pool = self.short_term_memory.pool.detach()

    def get_all(self, batch_size):
        if self.long_term_memory[0].queue_q is None:
            return self.short_term_memory.get_all(batch_size)
        return torch.cat(
            [memory.get_all(batch_size) for memory in self.long_term_memory[::-1]] + [self.short_term_memory.get_all(batch_size)],
            dim=1,
        )

    def get_long_term_memory(self, batch_size):
        all_q = []
        all_k = []
        all_v = []

        for memory in self.long_term_memory[::-1]:
            q, k, v = memory.get_all(batch_size)
            if q is not None and k is not None and v is not None:
                all_q.append(q)
                all_k.append(k)
                all_v.append(v)

        if not all_q or not all_k or not all_v:
            return None, None, None

        return torch.cat(all_q, dim=1), torch.cat(all_k, dim=1), torch.cat(all_v, dim=1)

    def clear_all(self):
        for memory in self.long_term_memory:
            memory.clear()
        self.short_term_memory.clear()

