import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, fields=None):
        """

        Args:
            dataset: datasets.arrow_dataset.Dataset
            tokenizer:
        """

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.fields = fields

        # 删除回答为空的样本
        self.dataset = self.dataset.filter(lambda x: x[self.fields[2]] is not None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]

        system = {"role": "system", "content": row[self.fields[0]] if row[self.fields[0]] is not None else ""}
        question = {"role": "user", "content": row[self.fields[1]] if row[self.fields[1]] is not None else ""}
        response = {"role": "assistant", "content": row[self.fields[2]] if row[self.fields[2]] is not None else ""}

        text = self.tokenizer.apply_chat_template([system, question, response], tokenize=False, add_special_tokens=False)

        sq_text = self.tokenizer.apply_chat_template([system, question], tokenize=False, add_special_tokens=False)

        """
        记录一下, 这里如果system为空, 会导致tokenizer自行增加一段文本, 导致长度不一致
        比如如果system为空, tokenizer会自动增加 "You are a helpful assistant."
        因此, answer需要通过text去掉前面system_question的长度来获取
        """
        a_text = text[len(sq_text):]

        # 输入去掉第最后一个token
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)[:-1]
        # 输出去掉第一个token
        output_ids = self.tokenizer.encode(text, add_special_tokens=False)[1:]

        # system_question和answer及其长度
        sq_ids = self.tokenizer.encode(sq_text, add_special_tokens=False)
        system_question_len = len(sq_ids)
        a_ids = self.tokenizer.encode(a_text, add_special_tokens=False)
        answer_len = len(a_ids)
        # 计算整体的长度
        text_len = len(input_ids) + 1
        assert text_len == system_question_len + answer_len, f"{text_len} != {system_question_len} + {answer_len}"

        return {
            'input_ids': input_ids,
            'output_ids': output_ids,
            'system_question_ids': sq_ids,
            'answer_ids': a_ids,
            'system_question_len': system_question_len,
            'answer_len': answer_len,
        }

    def sort(self, keys):
        for key in keys:
            self.dataset = sorted(self.dataset, key=lambda x: len(x[key]), reverse=False)

    def filter_by_length(self, max_length, keys):
        # 过滤 keys 字段长度之和超过max_length的样本 同时避免出现空样本
        self.dataset = self.dataset.filter(
            lambda x:
            sum(
                len(self.tokenizer.encode(x[key], add_special_tokens=False)) if x[key] is not None else 0
                for key in keys
            ) <= max_length
        )


# 定义自定义collate_fn
def collate_fn(batch, tokenizer):
    batch_input_ids = [item['input_ids'] for item in batch]
    batch_output_ids = [item['output_ids'] for item in batch]
    question_lengths = [item['system_question_len'] for item in batch]

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

        if 1 in mask:  # Skip samples where mask doesn't have any 1s
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


def pretraining_get_batch(config, split, context_len, device, device_type, validation=False):
    # randomize the context length from 0 to context_len
    # if not validation:
    #     context_len = np.random.randint(1, context_len + 1)
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


def get_batch(config, device, device_type, split='train', data_iter=None, validation=False):
    if config.train_mode == 'pretrain':
        X, Y = pretraining_get_batch(
            config,
            split,
            config.train_size if split == 'train' else config.val_size,
            device,
            device_type,
            validation=validation,
        )  # fetch the very first batch
        masks = None
    elif config.train_mode == 'sft':
        input_ids, output_ids, masks = next(data_iter)
        X = input_ids.to(device)
        Y = output_ids.to(device)
    else:
        raise ValueError(f"Invalid train_mode: {config.train_mode}")
    return X, Y, masks


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch
