import torch
from tqdm import tqdm

from dataloader import get_batch


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(config, model, ctx, device, device_type, iter_num):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in tqdm(range(config['eval_iters']), desc=f"Evaluating {split} loss"):
            context_len = config['train_size'] if split == 'train' else config['val_size']
            # 如果iter_num可以被1000整除，则将context_len翻n倍
            if iter_num > 0 and iter_num % 10000 == 0 and split == 'val':
                context_len *= 16
            input_x, label_y = get_batch(
                config,
                split,
                context_len=context_len,
                device=device,
                device_type=device_type,
            )
            with ctx:
                # # Calculate the time it takes to evaluate the loss
                # import time
                # start = time.time()
                _, loss = model(input_x, label_y)
                # print(f"Time taken to evaluate loss: {time.time() - start}")
            losses[k] = loss.item()
        out[split] = losses.mean()
        # 计算困惑度
        out[f'{split}_perplexity'] = torch.exp(out[split])
    model.train()
    return out
