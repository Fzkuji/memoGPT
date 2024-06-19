import time
from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    max_batch_size: int = 64

    short_term_memory_size: int = 16
    long_term_memory_layer: int = 16
    long_term_memory_chunk_size: int = 4
    long_term_memory_size = ([short_term_memory_size * long_term_memory_chunk_size] * (long_term_memory_layer - 1) +
                             [short_term_memory_size * (long_term_memory_chunk_size - 1)])

    rope_theta: float = 500000
    rms_norm_eps: float = 1e-6

    input_block_size: int = 1024
    memory_block_size: int = 256

    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency

    n_layer: int = 12
    num_attention_heads: int = 14
    num_key_value_heads: int = 2

    use_moe: bool = False
    n_expert: int = 16
    n_expert_per_tok: int = 4
    n_embd: int = 896
    intermediate_size: int = 4864

    hidden_act = "silu"

    dropout: float = 0.1
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device: str = 'cuda'

    init_from: str = 'Qwen/Qwen2-0.5B-Instruct'


@dataclass
class TrainConfig(GPTConfig):

    seed: int = 1337

    config_file: str = 'configs/finetune_gpt2.py'

    # 输出和日志
    out_dir: str = 'out-owt'
    eval_interval: int = 500
    eval_iters: int = 20
    eval_only: bool = False

    log_interval: int = 10
    wandb_log: bool = False
    wandb_project: str = 'owt'
    wandb_run_name: str = 'ft-' + str(time.time())

    # 数据和初始化
    dataset: str = 'openwebtext'
    train_mode: str = 'sft'
    init_from: str = 'Qwen/Qwen2-0.5B-Instruct'
    data_dir: str = 'data'

    # 检查点设置
    always_save_checkpoint: bool = True

    # 训练参数
    train_size_ratio: int = 2
    val_size_ratio: int = 2
    train_size: int = 2
    val_size: int = 2

    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_iters: int = 600000
    lr_decay_iters: int = 10000
    warmup_iters: int = 200

    # 模型参数（GPTConfig 已经有的参数不重复）
    memory_dim: int = 768

    # 优化器参数
    learning_rate: float = 4e-5
    decay_lr: bool = True
    min_lr: float = 1e-6

    # AdamW 优化器参数
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # 分布式训练设置
    backend: str = 'nccl'

    # 系统设置
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = False



