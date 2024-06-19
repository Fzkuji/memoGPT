import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        question = row['question']
        response = row['response']

        input_text = question + response
        output_text = question + response + ' ' + self.tokenizer.eos_token

        input_ids = self.tokenizer.encode(input_text)
        output_ids = self.tokenizer.encode(output_text)[1:]  # 去掉第一个token

        return {
            'input_ids': input_ids,
            'output_ids': output_ids,
            'question_len': len(self.tokenizer.encode(question))
        }


# 定义自定义collate_fn
def collate_fn(batch, tokenizer):
    batch_input_ids = [item['input_ids'] for item in batch]
    batch_output_ids = [item['output_ids'] for item in batch]
    question_lengths = [item['question_len'] for item in batch]

    max_len = max(max(len(ids) for ids in batch_input_ids), max(len(ids) for ids in batch_output_ids))

    input_ids_padded = []
    output_ids_padded = []
    masks = []

    for input_ids, output_ids, q_len in zip(batch_input_ids, batch_output_ids, question_lengths):
        input_len = len(input_ids)
        output_len = len(output_ids)

        # Padding input_ids and output_ids to the same length
        input_ids += [tokenizer.pad_token_id] * (max_len - input_len)
        output_ids += [tokenizer.pad_token_id] * (max_len - output_len)

        # Create mask: 0 for question part, 1 for response part, 0 for padding and eos_token part
        mask = [0] * q_len + [1] * (output_len - q_len) + [0] * (max_len - output_len)

        input_ids_padded.append(input_ids)
        output_ids_padded.append(output_ids)
        masks.append(mask)

    input_ids_padded = torch.tensor(input_ids_padded, dtype=torch.long)
    output_ids_padded = torch.tensor(output_ids_padded, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.long)

    return input_ids_padded, output_ids_padded, masks


'''
# 加载数据集
dataset = load_dataset("Open-Orca/OpenOrca", split="train")

# 初始化Llama的tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 定义常量
END_OF_TEXT_TOKEN = tokenizer.eos_token

# 创建数据集和DataLoader
custom_dataset = CustomDataset(dataset, tokenizer)
dataloader = DataLoader(custom_dataset, batch_size=2, collate_fn=lambda x: collate_fn(x, tokenizer))

# 示例：获取一个batch的数据
for batch in dataloader:
    input_ids, output_ids, masks = batch
    print("Input IDs:", input_ids)
    print("Output IDs:", output_ids)
    print("Masks:", masks)
    break
'''


def pretraining_get_batch(config, split, context_len, device, device_type):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - context_len, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + context_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + context_len]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_batch(config, device, device_type, data_iter=None):
    if config.train_mode == 'pretrain':
        X, Y = pretraining_get_batch(
            config,
            'train',
            config.train_size,
            device,
            device_type
        )  # fetch the very first batch
        masks = None
    elif config.train_mode == 'sft':
        input_ids, output_ids, masks = next(data_iter)
        X = input_ids.to(device)
        Y = output_ids.to(device)
    else:
        raise ValueError(f"Invalid train_mode: {config.train_mode}")
    return X, Y, masks
