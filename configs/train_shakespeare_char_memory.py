# Memory Transformer + RoPE


# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'output/out-memory-gpt-shakespeare-char'
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64

memory_size = 4
block_size = 64  # context of up to 256 previous characters
train_size = block_size * 16
val_size = block_size * 16
# memory_size = 32

# 这个值挺好的
# memory_size = 16
# block_size = 128  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
use_moe = False
n_expert = 8
n_expert_per_tok = 3
memory_dim = 384
n_embd = memory_dim
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 1e-7  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
