"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN

from . import GPTConfig
from models.networks import LayerNorm, FeedForward, MoeArgs, MoeLayer
from models.attentions.MemorySelfAttention import MemorySelfAttention


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        # 用config类型判断是否使用MemorySelfAttention
        self.self_attn = MemorySelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, short_term_memory_init=False, update_memory=False):
        x = x + self.self_attn(self.input_layernorm(x), short_term_memory_init, update_memory)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.input_block_size is not None
        self.config = config
        # print("configs.vocab_size: ", configs.vocab_size)
        self.model = nn.ModuleDict(dict(
            embed_tokens=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            layers=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm=RMSNorm(config.n_embd, eps=config.rms_norm_eps),

        ))
        # 输出wte支持的最大输入
        print("prediction vocabulary size: ", self.model.embed_tokens.num_embeddings)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.past_input = None
        self.tokens_left_to_update_memory = self.config.input_block_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.init_from,
            # cache_dir='.cache/huggingface/hub',
        )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None, cal_segment_loss=False, **kwargs):
        # torch.autograd.set_detect_anomaly(True)
        # print("New forwarding...Input length: ", idx.size(1))

        batch_size, input_len = input_ids.size()

        # 初始化短期记忆
        if self.model.layers[0].self_attn.memory.short_term_memory.pool is None:

            print("Initializing short-term memory...")

            # 创建一个初始化输入idx，值都是<|im_start|>，也就是151644
            short_term_memory_init_idx = torch.full((batch_size, self.config.short_term_memory_size),
                                                    self.tokenizer.eos_token_id, dtype=torch.long,
                                                    device=input_ids.device)



            # 将初始化输入idx传入模型
            output = self.model.embed_tokens(short_term_memory_init_idx)

            # 初始化短期记忆
            for block in self.model.layers:
                output = block(output, short_term_memory_init=True)

            # print("Short-term memory initialized.")

        # 定义 input_block_size 和 memory_block_size
        input_block_size = self.config.input_block_size  # 例如 1024
        memory_block_size = self.config.memory_block_size  # 例如 256
        # print("input_block_size: ", input_block_size)
        # print("memory_block_size: ", memory_block_size)

        if self.past_input is not None:
            end_pos = self.past_input.size(1)
            input_ids = torch.cat((self.past_input, input_ids), dim=1)
            self.past_input = None
        else:
            end_pos = 0

        # 嵌入所有的 tokens
        tok_emb = self.model.embed_tokens(input_ids)

        # 准备变量
        logits = None
        seq_len = tok_emb.size(1)  # 序列总长度
        # print("seq_len (length of the whole seq): ", seq_len)

        # 如果是训练 targets不为空 则创建logits保存预测结果
        if labels is not None:
            logits = torch.zeros((batch_size, input_len, self.config.vocab_size), device=tok_emb.device,
                                 dtype=self.config.torch_dtype)

        output_start_pos = 0
        while end_pos < seq_len:

            # print("forwarding...")

            if self.tokens_left_to_update_memory - (seq_len - end_pos) > 0:
                # 如果输入序列的长度小于更新记忆需要的 token 数
                self.tokens_left_to_update_memory -= (seq_len - end_pos)
                predict_len = seq_len - end_pos
                end_pos = seq_len
                start_pos = max(end_pos - input_block_size - memory_block_size, 0)
                memory_update_flag = False
            else:
                # 如果输入序列的长度大于更新记忆需要的 token 数
                predict_len = self.tokens_left_to_update_memory
                end_pos = end_pos + self.tokens_left_to_update_memory
                start_pos = max(end_pos - input_block_size - memory_block_size, 0)
                self.tokens_left_to_update_memory = memory_block_size
                memory_update_flag = True

            # 获取当前块的 tok_emb
            output = tok_emb[:, start_pos:end_pos, :]
            # print("output shape: ", output.shape)

            # print("memory_update_flag: ", memory_update_flag)
            # 通过模型层
            for block in self.model.layers:
                output = block(output, update_memory=memory_update_flag)

            # 获取当前块的 logits
            # if end_pos == seq_len:

            # 当targets不为空且当前块的mask有1时，才保存logits
            if labels is not None:
                if attention_mask is None or attention_mask[:,
                                             output_start_pos:output_start_pos + predict_len].sum() > 0:
                    output = self.lm_head(self.model.norm(output))
                    logits[:, output_start_pos:output_start_pos + predict_len, :] = output[:, -predict_len:, :]
            else:
                logits = self.lm_head(self.model.norm(output))[:, [-1], :]
            output_start_pos += predict_len

        if labels is not None:
            # 清除模型层的记忆
            self.clear_memory()

            if attention_mask is not None:
                labels[attention_mask == 0] = -1  # 将被 mask 的位置设置为 -1
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)

            loss = F.cross_entropy(logits, labels, ignore_index=-1)

            # 计算 segment_loss
            if cal_segment_loss:
                segment_loss = []

                # 不确定什么长度，每个 memory_block_size 长度计算一次 loss
                for i in range(0, input_len, memory_block_size):
                    segment_loss.append(
                        F.cross_entropy(logits[i:i + memory_block_size, :], labels[i:i + memory_block_size],
                                        ignore_index=-1))

                # concert segment_losses to tensor
                segment_loss = torch.stack(segment_loss)

            else:
                segment_loss = None

            return logits, loss, segment_loss if cal_segment_loss else None,
        else:
            # save the past input for next iteration
            pos = max(seq_len - self.config.memory_block_size - self.config.input_block_size + 1, 0)
            self.past_input = input_ids[:, pos:]

            logits = logits[:, [-1], :]  # 只保留最后一个时间步的 logits
            return logits, None

    def clear_memory(self):
        for block in self.model.layers:
            block.self_attn.memory.clear_all()
        self.past_input = None
        self.tokens_left_to_update_memory = self.config.input_block_size

    @classmethod
    def from_pretrained(cls, model_type, override_args):
        from transformers import AutoModelForCausalLM
        print(f"loading weights from pretrained {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'Qwen/Qwen2-0.5B-Instruct': dict(n_layer=24, num_attention_heads=14, num_key_value_heads=2, n_embd=896, intermediate_size=4864, vocab_size=151936, torch_dtype=torch.bfloat16, rms_norm_eps=1e-06, bias=True),
            'Qwen/Qwen2-1.5B-Instruct': dict(n_layer=28, num_attention_heads=12, num_key_value_heads=2, n_embd=1536, intermediate_size=8960, vocab_size=151936, torch_dtype=torch.bfloat16, rms_norm_eps=1e-06, bias=True),
            'Qwen/Qwen2-7B-Instruct':   dict(n_layer=28, num_attention_heads=28, num_key_value_heads=4, n_embd=3584, intermediate_size=18944, vocab_size=152064, torch_dtype=torch.bfloat16, rms_norm_eps=1e-06, bias=True),
            'Qwen/Qwen2-7B':            dict(n_layer=28, num_attention_heads=28, num_key_value_heads=4, n_embd=3584, intermediate_size=18944, vocab_size=152064, torch_dtype=torch.bfloat16, rms_norm_eps=1e-06, bias=True),
            'meta-llama/Llama-2-7b-chat-hf': dict(n_layer=32, num_attention_heads=32, num_key_value_heads=32,
                n_embd=4096, hidden_size=4096, intermediate_size=11008, vocab_size=32000, torch_dtype=torch.float16, rms_norm_eps=1e-05,
                bias=False, rope_theta=10000.0, rope_scaling=None),
            'princeton-nlp/Sheared-LLaMA-1.3B': dict(n_layer=24, num_attention_heads=16, num_key_value_heads=16, n_embd=2048, hidden_size=2048, intermediate_size=5504, vocab_size=32000, torch_dtype=torch.float32, rms_norm_eps=1e-05, bias=False, rope_theta=10000.0, rope_scaling=None),
            'meta-llama/Meta-Llama-3.1-8B-Instruct': dict(n_layer=32, num_attention_heads=32, num_key_value_heads=8, n_embd=4096, intermediate_size=14336, vocab_size=128256, torch_dtype=torch.bfloat16, rms_norm_eps=1e-05, bias=False),
        }[model_type]
        config_args['model_type'] = model_type  # always True for GPT model checkpoints
        # add all args from override_args to config_args
        override_args.update(config_args)

        # 过滤掉不需要的参数
        override_args = {k: v for k, v in override_args.items() if k in GPTConfig.__dataclass_fields__.keys()}

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**override_args)
        model = GPT(config)

        # change model dtype to torch_dtype
        # model = model.to(dtype)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.self_attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.self_attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.self_attn.bias')]  # same, just the mask (buffer)

        """Test Parameters (For Debugging)
        Find and print the mismatched key names
        """
        # print(sd_keys)
        # print(sd_keys_hf)
        #
        # # fine keys in sd that are not in sd_hf
        # missing_keys = [k for k in sd_keys if k not in sd_keys_hf]
        # print(f"missing keys: {missing_keys}")
        #
        # # find keys in sd_hf that are not in sd
        # extra_keys = [k for k in sd_keys_hf if k not in sd_keys]
        # print(f"extra keys: {extra_keys}")

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        """
        Copy Parameters
        """

        for k in sd_keys_hf:
            # vanilla copy over the other parameters
            # print shape mismatches
            if sd[k].shape != sd_hf[k].shape:
                print(f"shape mismatch: {k} shape {sd[k].shape} != {sd_hf[k].shape}")
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
        print("loaded successfully")
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.num_attention_heads, cfg.n_embd // cfg.num_attention_heads, cfg.input_block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 119e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=400, eos_token_id=None, temperature=0.3, top_k=None, output_type="str",
                 **kwargs):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence indefinitely, feeding the predictions back into the model each time.
        The input sequence is divided into blocks of size input_block_size and memory is updated accordingly.
        Stop generating if the eos_token_id is generated.
        """
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained(
            self.config.init_from,
            # add_special_tokens=False,
        )

        # 判断输入idx是否为字符串
        if isinstance(input_ids, str):
            input_ids = enc.encode(input_ids, add_special_tokens=False)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.config.device)

        if eos_token_id is None:
            eos_token_id = enc.convert_tokens_to_ids(enc.pad_token)

        idx_cond = input_ids

        # print("idx shape: ", idx.shape)

        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_cond = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_cond), dim=1)

            # If the generated token is the eos_token_id, stop generating
            if idx_cond.item() == eos_token_id:
                break

        if output_type == "str":
            return enc.decode(input_ids[0])
        elif output_type == "idx":
            return input_ids
        else:
            raise ValueError("output_type must be 'str' or 'idx'")

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        print("prepare_inputs_for_generation")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
