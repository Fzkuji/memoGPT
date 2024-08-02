import torch
import torch.nn as nn

from models.memoryGPT.config import GPTConfig


class MemoryPool(nn.Module):
    """ A simple pool for storing tensors with a fixed capacity """

    def __init__(self, config, capacity, *tensor_dims, max_batch_size=64):
        super(MemoryPool, self).__init__()
        self.pool = None
        self.config = config
        self.batch_size = max_batch_size
        self.capacity = capacity
        self.tensor_dims = tensor_dims

    def get_all(self, batch_size):
        """ Return a tensor containing all elements in the pool """
        assert batch_size <= self.batch_size, f"Batch size {batch_size} is greater than the maximum batch size {self.batch_size}"
        return self.pool[:batch_size]
        # return self.pool_q[:batch_size], self.pool_k[:batch_size], self.pool_v[:batch_size]

    def get_len(self):
        return self.capacity

    def init(self, pool):
        self.pool = pool.to(self.config.device)

    def update(self, tensor):
        """ Update the pool with a new tensor """
        # assert tensor_q.shape == tensor_k.shape == tensor_v.shape, "All tensors should have the same shape"

        bsz, seqlen, *dim = tensor.shape
        assert bsz <= self.batch_size, f"Batch size {bsz} is greater than the maximum batch size {self.batch_size}"
        assert seqlen == self.capacity, f"Sequence length {seqlen} is not equal to the capacity {self.capacity}"

        self.pool[:bsz, :, :] = tensor.detach()

    def clear(self):
        """ Clear the pool """
        self.pool = None


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
        self.queue = []  # Max: torch.zeros(max_batch_size, capacity, *tensor_dims)
        # self.queue_v = []
        self.index = 0

    def push(self, tensor):
        if len(self.queue) == 0:
            self.batch_size = tensor.shape[0]

        """ Add a tensor to the queue """
        bsz, seqlen, *dim = tensor.shape  # bsz: batch size, seqlen: sequence length
        if bsz != self.batch_size:
            self.clear()
            print(f"Long-term memory cleared. Batch size changed from {self.batch_size} to {bsz}.")

        self.queue.append(tensor.detach())
        # self.queue_v.append(tensor_v.detach())

        if len(self.queue) > self.capacity:
            self.queue.pop(0)
            # self.queue_v.pop(0)
            self.index += 1

        if self.index > self.capacity:
            self.index = self.index - self.capacity
            return True  # Carry over
        else:
            return False

    def get_all(self, batch_size=0):
        """ Return a tensor containing all elements in the queue """
        if self.queue and batch_size != self.queue[0].shape[0]:
            self.clear()
        return self.queue

    def get_len(self):
        return len(self.queue)

    def clear(self):
        """ Clear the queue """
        self.queue = []
        # self.queue_v = []
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

    def update_long_term_memory(self, tensor):
        # for memory in self.long_term_memory:
        #     memory.update_rotary_emb(freqs_cis)
        for memory in self.long_term_memory:
            if tensor is not None:
                carry_over = memory.push(tensor.detach())
                if not carry_over:
                    break
        torch.cuda.empty_cache()

    def init_short_term_memory(self, tensor):
        self.short_term_memory.init(tensor.detach())
        torch.cuda.empty_cache()

    def update_short_term_memory(self, tensor):
        self.short_term_memory.update(tensor.detach())
        torch.cuda.empty_cache()

    def get_long_term_memory(self, batch_size):
        long_term_memories = []
        # all_v = []

        for memory in self.long_term_memory[::-1]:
            k = memory.get_all(batch_size)
            long_term_memories.extend(k)
            # all_v.extend(v)

        if len(long_term_memories) == 0:
            return None
        else:
            # 得用concatenate 否则维度会变化
            long_term_memories = torch.cat(long_term_memories, dim=1)
            return long_term_memories

    def get_short_term_memory(self, batch_size):
        return self.short_term_memory.get_all(batch_size)

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
