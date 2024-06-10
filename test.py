import torch


def extend_causal_mask(original_mask, long_memory_length, short_memory_length):
    original_length = original_mask.size(-1)
    new_length = long_memory_length + short_memory_length + original_length

    # 创建新的 mask，初始化为 0
    new_mask = torch.zeros((original_mask.size(0), original_mask.size(1), new_length, new_length),
                           device=original_mask.device)

    # 设置短期记忆部分
    new_mask[:, :, :, :long_memory_length + short_memory_length] = 0  # 前 long_memory_length + short_memory_length 列

    # 设置长期记忆部分
    new_mask[:, :, :long_memory_length, :] = -float('inf')  # 前 long_memory_length 行

    # 将原始 mask 放到新的位置
    new_mask[:, :, long_memory_length + short_memory_length:, long_memory_length + short_memory_length:] = original_mask

    return new_mask


# 示例使用
original_mask = torch.tensor([
    [0.0000e+00, -3.3895e+38, -3.3895e+38, -3.3895e+38],
    [0.0000e+00, 0.0000e+00, -3.3895e+38, -3.3895e+38],
    [0.0000e+00, 0.0000e+00, 0.0000e+00, -3.3895e+38],
    [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]
]).unsqueeze(0).unsqueeze(0)  # 增加 batch 和 head 维度

long_memory_length = 2
short_memory_length = 2

new_mask = extend_causal_mask(original_mask, long_memory_length, short_memory_length)

# 使用 torch.set_printoptions 来控制输出格式
torch.set_printoptions(profile="full", linewidth=1000)
print(new_mask)