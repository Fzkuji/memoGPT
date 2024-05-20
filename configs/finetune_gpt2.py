import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False  # feel free to turn on
wandb_project = 'owt'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'openwebtext'
init_from = 'scratch'  # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 3
gradient_accumulation_steps = 1
max_iters = 20

memory_dim = 768
n_embd = 768
short_term_memory_size = 16

block_size = 512  # context of up to 256 previous characters
train_size = block_size * 16
val_size = block_size * 16

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
