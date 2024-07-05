import torch
from tqdm import tqdm

from dataloader import pretraining_get_batch, get_batch


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(config, model, ctx, device, device_type, iter_num, dataiter=None):
    if config.train_mode == 'pretrain':
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_iters)
            if split == 'train':
                segment_losses = torch.zeros(config.train_size_ratio).to(device)
            else:
                segment_losses = torch.zeros(config.val_size_ratio).to(device)

            for k in tqdm(range(config.eval_iters), desc=f"Evaluating {split} loss"):

                input_x, label_y, masks = get_batch(config, device, device_type, split=split, data_iter=dataiter, validation=True)

                with ctx:
                    # # Calculate the time it takes to evaluate the loss
                    # import time
                    # start = time.time()
                    _, loss, segment_loss = model(
                        input_x,
                        label_y,
                        attention_mask=masks,
                        cal_segment_loss=True,
                    )
                    torch.cuda.empty_cache()  # 清除未使用的缓存内存
                    # print(f"Time taken to evaluate loss: {time.time() - start}")
                    segment_losses += segment_loss
                losses[k] = loss.item()
            out[split] = losses.mean()
            # 计算困惑度
            out[f'{split}_perplexity'] = torch.exp(out[split])
            out[f'{split}_segment_loss'] = segment_losses/config.eval_iters
        model.train()
        return out
    elif config.train_mode == 'sft':
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_iters)
            # if split == 'train':
            #     segment_losses = torch.zeros(config.train_size_ratio).to(device)
            #     segment_count = torch.zeros(config.train_size_ratio).to(device)
            # else:
            #     segment_losses = torch.zeros(config.val_size_ratio).to(device)
            #     segment_count = torch.zeros(config.val_size_ratio).to(device)

            for k in tqdm(range(config.eval_iters), desc=f"Evaluating {split} loss"):
                input_x, label_y, masks = get_batch(config, device, device_type, split=split, data_iter=dataiter,
                                                    validation=True)

                with ctx:
                    # # Calculate the time it takes to evaluate the loss
                    # import time
                    # start = time.time()
                    _, loss, segment_loss = model(
                        input_x,
                        label_y,
                        attention_mask=masks,
                        cal_segment_loss=True,
                    )
                    torch.cuda.empty_cache()  # 清除未使用的缓存内存
                    # print(f"Time taken to evaluate loss: {time.time() - start}")
                    # 确定segment_loss的大小
                    # segment_losses += segment_loss

                    # 不确定segment_loss的长度，因此segment_loss加到segment_losses时，取平均时需要考虑相加的次数
                    # 这里应该考虑segment_loss的长度，如果长度为5，则segment_count前5个元素都要加1
                    # 同时将segment_loss加到segment_losses的前5个元素上
                    # segment_count[:segment_loss.size(0)] += 1
                    # segment_losses[:segment_loss.size(0)] += segment_loss

                    # print(f"segment_loss: {segment_loss}")
                    # print(f"segment_losses: {segment_losses}")

                losses[k] = loss.item()
            out[split] = losses.mean()
            # 计算困惑度
            out[f'{split}_perplexity'] = torch.exp(out[split])
            # out[f'{split}_segment_loss'] = segment_losses / segment_count
            out[f'{split}_segment_loss'] = None
        model.train()
        return out

