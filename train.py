import os
import time
import pickle
import math
from contextlib import nullcontext
from dataclasses import fields

import torch
from datasets import load_dataset, DatasetDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.utils import get_lr
from dataloader import pretraining_get_batch, CustomDataset, collate_fn, get_batch
from models.memoryGPT.eval import estimate_loss
from models.memoryGPT.gpt2 import GPT
from models.memoryGPT.config import GPTConfig, TrainConfig


# 从配置文件加载配置
config_file = 'configs/finetune_gpt2.py'
config_vars = {}
with open(config_file, 'r', encoding='utf-8') as f:
    exec(f.read(), {}, config_vars)

# 将配置文件中的所有变量加载到config对象中
config_dict = {k: v for k, v in config_vars.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}
train_config_fields = {field.name for field in fields(TrainConfig)}
filtered_config_dict = {k: v for k, v in config_dict.items() if k in train_config_fields}
config = TrainConfig(**filtered_config_dict)

# 现在可以使用 config.参数名 来访问配置了
print(config)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    print("using distributed data parallel")
    init_process_group(backend=config.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    print("not using distributed data parallel")
    device = config.device
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.input_block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

config_dict['data_dir'] = os.path.join('data', config.dataset)
config.data_dir = config_dict['data_dir']
print(f"load data from {config_dict['data_dir']}")

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(config_dict['data_dir'], 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init choose arguments from config_dict that GPTConfig has
model_args = {k: v for k, v in config_dict.items() if k in GPTConfig.__dataclass_fields__}
if config.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif config.init_from.startswith('Qwen') or config.init_from.startswith('meta'):
    print(f"Initializing from {config.init_from} weights")
    model = GPT.from_pretrained(config.init_from, config_dict)
    # read off the created configs params, so we can store them into checkpoint correctly
    model_args = {k: getattr(model.config, k) for k in GPTConfig.__dataclass_fields__}
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # create the model
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Unsupported init_from: {config.init_from}")

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
if config.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if config.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# logging
if config.wandb_log and master_process:
    import wandb

    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config_dict)

# Data loading

if config.train_mode == 'pretrain':
    train_iter = None
    val_iter = None

elif config.train_mode == 'sft':
    # 加载数据集
    dataset = load_dataset("Open-Orca/OpenOrca", split="train")
    train_valtest = dataset.train_test_split(test_size=0.2, seed=config.seed)
    val_test = train_valtest['test'].train_test_split(test_size=0.5, seed=config.seed)
    dataset = DatasetDict({
        'train': train_valtest['train'],
        'val': val_test['train'],
        'test': val_test['test'],
    })
    # 初始化Llama的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    # 创建数据集和DataLoader
    train_dataset = CustomDataset(dataset['train'], tokenizer)
    val_dataset = CustomDataset(dataset['val'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=lambda x: collate_fn(x, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=lambda x: collate_fn(x, tokenizer))

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

else:
    raise ValueError(f"Invalid train_mode: {config.train_mode}")

X, Y, masks = get_batch(config, device, device_type, data_iter=train_iter)

t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, config.warmup_iters, config.lr_decay_iters, config.learning_rate,
                config.min_lr) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss(
            config,
            model,
            ctx,
            device,
            device_type,
            iter_num,
            dataiter=val_iter if config.train_mode == 'sft' else None
        )
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity {losses['train_perplexity']:.4f}, val perplexity {losses['val_perplexity']:.4f}")
        if config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/perplexity": losses['train_perplexity'],
                "val/perplexity": losses['val_perplexity'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        if losses['val'] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'configs': config_dict,
                }
                print(f"saving checkpoint to {config.out_dir}")
                if losses['val'] < best_val_loss:
                    torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
                elif config.always_save_checkpoint:
                    torch.save(checkpoint, os.path.join(config.out_dir, f'{iter_num}.pt'))
    if iter_num == 0 and config.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, Y, masks=masks)
            loss = loss / config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, masks = get_batch(config, device, device_type, data_iter=train_iter)

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > config.max_iters:
        break

if ddp:
    destroy_process_group()