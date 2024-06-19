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

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # print("configs.vocab_size: ", configs.vocab_size)
        self.model = nn.ModuleDict(dict(
            embed_tokens=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            layers=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm=RMSNorm(config.n_embd, eps=config.rms_norm_eps),

        ))
        # 输出wte支持的最大输入
        print("wte max: ", self.model.embed_tokens.num_embeddings)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

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

    def split_sequence(self, input_sequence, section_length):
        """
        Split a multi-dimensional input sequence into smaller sections along the time dimension.

        Args:
        - input_sequence (torch.Tensor): Input sequence tensor of shape (b, t, n_embd).
        - section_length (int): Length of each section along the time dimension.

        Returns:
        - List[torch.Tensor]: List of tensors, each representing a section of the input sequence.
        """
        b, t, n_embd = input_sequence.size()
        num_sections = math.ceil(t / section_length)
        sections = []

        for i in range(num_sections):
            start_idx = i * section_length
            end_idx = min((i + 1) * section_length, t)
            section = input_sequence[:, start_idx:end_idx, :]
            sections.append(section)

        return sections

    def forward(self, idx, targets=None, masks=None, index=None):
        tok_emb = self.model.embed_tokens(idx)
        del idx  # 删除不再使用的变量
        torch.cuda.empty_cache()  # 清除未使用的缓存内存

        tok_emb_sections = self.split_sequence(tok_emb, self.config.input_block_size)
        section_len = tok_emb_sections[0].size(1)
        del tok_emb  # 删除不再使用的变量
        torch.cuda.empty_cache()  # 清除未使用的缓存内存

        losses = []
        current_pos = 0
        logits = None

        if index is not None:  # 如果index不为None，则只计算指定index段的的损失和预测
            if index == -1:
                index = len(tok_emb_sections) + index
            assert 0 <= index < len(tok_emb_sections)

            # 输入index段之前的内容，初始化memory
            current_index = 0
            with torch.no_grad():
                for tok_emb_section in tok_emb_sections:
                    if current_index == index:
                        break
                    for block in self.model.layers:
                        tok_emb_section = block(tok_emb_section)

                    del tok_emb_section  # 删除不再使用的变量
                    torch.cuda.empty_cache()

                    current_pos += section_len
                    current_index += 1

            # 计算指定index段
            tok_emb_section = tok_emb_sections[index]
            for block in self.model.layers:
                tok_emb_section = block(tok_emb_section)

            logits_chunk = self.lm_head(self.model.norm(tok_emb_section))
            del tok_emb_section  # 删除不再使用的变量
            torch.cuda.empty_cache()

            # 计算 loss
            loss = F.cross_entropy(logits_chunk.reshape(-1, logits_chunk.size(-1)),
                                   targets[:, current_pos:current_pos + section_len].reshape(-1), ignore_index=-1)

            # 清除memory
            for block in self.model.layers:
                block.self_attn.memory.clear_all()

            # 使用index仅限于训练，因此不需要保存logits
            return None, loss

        else:  # 如果index为None，则计算整个Input的损失和预测
            for tok_emb_section in tok_emb_sections:
                section_len = tok_emb_section.size(1)
                for block in self.model.layers:
                    tok_emb_section = block(tok_emb_section)

                logits = self.lm_head(self.model.norm(tok_emb_section)) if logits is None else \
                    torch.cat((logits, self.lm_head(self.model.norm(tok_emb_section))), dim=1)

                del tok_emb_section,  # 删除不再使用的变量
                torch.cuda.empty_cache()  # 清除未使用的缓存内存

                current_pos += section_len

            for block in self.model.layers:
                block.self_attn.memory.clear_all()

            if targets is not None:
                if masks is not None:
                    targets[masks == 0] = -1  # 将被 mask 的位置设置为 -1
                logits = logits.reshape(-1, logits.size(-1))
                # print("logits shape: ", logits.shape)  # logits shape:  torch.Size([126, 151936])
                targets = targets.reshape(-1)
                # print("targets shape: ", targets.shape)  # targets shape:  torch.Size([905])
                loss = F.cross_entropy(logits, targets, ignore_index=-1)
                return None, loss
            else:
                logits = logits[:, [-1], :]  # 只保留最后一个时间步的 logits
                return logits, None

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        if "gpt2" in model_type:
            pass
        elif "Qwen" in model_type:
            override_args = override_args or {}  # default to empty dict
            # only dropout can be overridden see more notes below
            assert all(k == 'dropout' for k in override_args)
            from transformers import Qwen2ForCausalLM
            print("loading weights from pretrained Qwen2: %s" % model_type)

            # n_layer, n_head and n_embd are determined from model_type
            config_args = {
                'Qwen/Qwen2-0.5B-Instruct': dict(n_layer=24, num_attention_heads=14, num_key_value_heads=2, n_embd=896,
                                                 intermediate_size=4864, vocab_size=151936),
                'Qwen/Qwen2-1.5B-Instruct': dict(n_layer=28, num_attention_heads=12, num_key_value_heads=2, n_embd=1536,
                                                 intermediate_size=8960, vocab_size=151936),
                'Qwen/Qwen2-7B-Instruct': dict(n_layer=28, num_attention_heads=28, num_key_value_heads=4, n_embd=3584,
                                               intermediate_size=18944, vocab_size=152064),
            }[model_type]
            print("forcing block_size=32768, bias=True")
            config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
            config_args['bias'] = True  # always True for GPT model checkpoints
            # we can override the dropout rate, if desired
            if 'dropout' in override_args:
                print(f"overriding dropout rate to {override_args['dropout']}")
                config_args['dropout'] = override_args['dropout']
            # create a from-scratch initialized minGPT model
            config = GPTConfig(**config_args)
            model = GPT(config)

            # # print all state_dict shape
            # for key in model.state_dict().keys():
            #     print(key, model.state_dict()[key].shape)

            sd = model.state_dict()
            sd_keys = sd.keys()
            sd_keys = [k for k in sd_keys if
                       not k.endswith('.self_attn.bias')]  # discard this mask / buffer, not a param

            # init a huggingface/transformers model
            model_hf = Qwen2ForCausalLM.from_pretrained(
                model_type,
            )
            sd_hf = model_hf.state_dict()

            # copy while ensuring all of the parameters are aligned and match in names and shapes
            sd_keys_hf = sd_hf.keys()
            sd_keys_hf = [k for k in sd_keys_hf if
                          not k.endswith('.self_attn.masked_bias')]  # ignore these, just a buffer
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.self_attn.bias')]  # same, just the mask (buffer)

            # # print mismatched keys
            # print("mismatched keys: ", [k for k in sd_keys if k not in sd_keys_hf])
            # print("mismatched keys: ", [k for k in sd_keys_hf if k not in sd_keys])

            assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

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
        L, H, Q, T = cfg.n_layer, cfg.num_attention_heads, cfg.n_embd // cfg.num_attention_heads, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 119e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=None, eos_token_id=None, temperature=1.0, top_k=None, output_type="str"):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence indefinitely, feeding the predictions back into the model each time.
        The input sequence is divided into blocks of size block_size and memory is updated accordingly.
        Stop generating if the eos_token_id is generated.
        """
        # 判断输入idx是否为字符串
        if isinstance(idx, str):
            from transformers import AutoTokenizer
            enc = AutoTokenizer.from_pretrained(
                self.config.init_from,
            )
            idx = enc.encode(idx, add_special_tokens=False)
            idx = torch.tensor(idx).unsqueeze(0).to(self.config.device)

        B, T = idx.size()
        block_size = self.config.block_size

        from transformers import AutoTokenizer
        if eos_token_id is None:
            enc = AutoTokenizer.from_pretrained(
                self.config.init_from,
            )
            eos_token_id = enc.convert_tokens_to_ids(enc.pad_token)

        # Initialize memory by processing the initial blocks except the last one
        current_pos = 0
        while current_pos + block_size < T:
            idx_cond = idx[:, current_pos:current_pos + block_size]
            logits, _ = self(idx_cond)
            current_pos += block_size

        # Process the last block with special strategy
        index = min(current_pos, T - block_size)
        index = max(0, index)
        idx_cond = idx[:, index:]
        m = idx_cond.size(1)
        index = m - 1
        gen_len = 0

        while True:
            # Generate one token at a time until the length exceeds block size
            if index >= block_size:
                index = index - block_size
                idx_cond = idx_cond[:, -(block_size - 1):]  # Keep the last block_size - 1 tokens

            # Forward the model to get the logits for the last index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1).to(self.config.device)

            # Append sampled index to the running sequence
            idx_cond = torch.cat((idx_cond, idx_next), dim=1)
            if idx_cond.size(1) > block_size:
                idx_cond = idx_cond[:, 1:]
            index += 1

            # # Print the generated token for demonstration purposes
            # print(idx_next.item())

            # If the generated token is the eos_token_id, stop generating
            if idx_next.item() == eos_token_id:
                break

            # Keep generating tokens indefinitely
            idx = torch.cat((idx, idx_next), dim=1)

            gen_len += 1

            if (max_new_tokens is not None and gen_len >= max_new_tokens) or gen_len > 1024:
                break

        if output_type == "str":
            from transformers import AutoTokenizer
            enc = AutoTokenizer.from_pretrained(
                self.config.init_from,
            )
            return enc.decode(idx[0].tolist())

        return idx
