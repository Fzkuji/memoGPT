import os

import numpy as np
import torch


def get_batch(config, split, context_len, device, device_type):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(config['data_dir'], 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(config['data_dir'], 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - context_len, (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i + context_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + context_len]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y