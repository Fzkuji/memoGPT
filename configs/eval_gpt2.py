import time

import torch

# 输出和日志
out_dir = 'out-owt'
eval_interval = 500
eval_iters = 100
eval_only = False

log_interval = 10
wandb_log = False  # Feel free to turn on
wandb_project = 'owt'
wandb_run_name = 'ft-' + str(time.time())

# 数据和初始化
dataset = 'fineweb'  # fineweb, shakespeare, openwebtext
train_mode = 'pretrain'  # pretrain, sft
init_from = 'resume'  # 'Qwen/Qwen2-0.5B-Instruct', 'resume'

# 检查点设置
always_save_checkpoint = False  # Only save checkpoints if the validation loss improves

# 训练参数
batch_size = 1
gradient_accumulation_steps = 16
max_iters = 600000
lr_decay_iters = 100000
warmup_iters = 200  # how many steps to warm up for

# 模型参数
memory_dim = 896
intermediate_size = 4864
n_layer = 24
n_embd = 896
num_attention_heads = 14
num_key_value_heads = 2

short_term_memory_size = 16
bias = True  # Do we use bias inside LayerNorm and Linear layers?
rms_norm_eps = 1e-06
block_size = 1024
input_block_size = 256
train_size_ratio = 36  # 32
val_size_ratio = 36  # Need 22GB per 1024 * 1024 tokens long context
train_size = input_block_size * train_size_ratio
val_size = input_block_size * val_size_ratio

# 优化器参数
learning_rate = 8e-5
decay_lr = True
min_lr = 1e-6

# 额外的模型配置
use_moe = False
n_expert = 16
n_expert_per_tok = 4
dropout = 0.0  # For pretraining 0 is good, for finetuning try 0.1+

# AdamW 优化器参数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # Clip gradients at this value, or disable if == 0.0

# 分布式训练设置
backend = 'nccl'  # 'nccl', 'gloo', etc.

# 系统设置
device = 'cuda'  # Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # Use PyTorch 2.0 to compile the model to be faster
