import time

out_dir = 'out-owt'
eval_interval = 1000
eval_iters = 20
log_interval = 10
wandb_log = False  # feel free to turn on
wandb_project = 'owt'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'openwebtext'
init_from = 'Qwen/Qwen2-0.5B-Instruct'  # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = True

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 2
gradient_accumulation_steps = 8
max_iters = 600000
lr_decay_iters = 600000

memory_dim = 768
n_embd = 768
short_term_memory_size = 16

bias = True  # do we use bias inside LayerNorm and Linear layers?

block_size = 1024
input_block_size = 256
train_size_ratio = 32  # 32
val_size_ratio = 256  # need 22GB per 1024 * 1024 tokens long context
train_size = input_block_size * train_size_ratio
val_size = input_block_size * val_size_ratio

# finetune at constant LR
learning_rate = 3e-5
decay_lr = True
min_lr = 1e-6
