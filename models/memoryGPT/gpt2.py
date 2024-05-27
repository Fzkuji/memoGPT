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

from . import GPTConfig
from models.networks import LayerNorm, MLP, FeedForward, MoeArgs, MoeLayer
from models.attentions.MemorySelfAttention import MemorySelfAttention


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 用config类型判断是否使用MemorySelfAttention
        self.attn = MemorySelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.use_moe:
            self.moe_arg = MoeArgs(num_experts=config.n_expert, num_experts_per_tok=config.n_expert_per_tok)
            self.mlp = MoeLayer(
                experts=[FeedForward(config=config) for _ in range(config.n_expert)],
                gate=nn.Linear(config.n_embd, config.n_expert, bias=False),
                moe_args=self.moe_arg,
            )
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # print("configs.vocab_size: ", configs.vocab_size)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        # 输出wte支持的最大输入
        print("wte max: ", self.transformer.wte.num_embeddings)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

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
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
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

    def forward(self, idx, targets=None, index=None):
        tok_emb = self.transformer.wte(idx)
        del idx  # 删除不再使用的变量
        torch.cuda.empty_cache()  # 清除未使用的缓存内存

        tok_emb_sections = self.split_sequence(tok_emb, self.config.block_size)
        section_len = tok_emb_sections[0].size(1)
        del tok_emb  # 删除不再使用的变量
        torch.cuda.empty_cache()  # 清除未使用的缓存内存

        losses = []
        current_pos = 0
        logits_chunk = None

        if index is not None:  # 如果index不为None，则只计算指定index段的的损失和预测

            # 输入index段之前的内容，初始化memory
            current_index = 0
            with torch.no_grad():
                for tok_emb_section in tok_emb_sections:
                    if current_index == index:
                        break
                    for block in self.transformer.h:
                        tok_emb_section = block(tok_emb_section)

                    del tok_emb_section  # 删除不再使用的变量
                    torch.cuda.empty_cache()

                    current_pos += section_len
                    current_index += 1

            # 计算指定index段
            tok_emb_section = tok_emb_sections[index]
            for block in self.transformer.h:
                tok_emb_section = block(tok_emb_section)

            logits_chunk = self.lm_head(self.transformer.ln_f(tok_emb_section))
            del tok_emb_section  # 删除不再使用的变量
            torch.cuda.empty_cache()

            # 计算 loss
            loss = F.cross_entropy(logits_chunk.reshape(-1, logits_chunk.size(-1)), targets[:, current_pos:current_pos + section_len].reshape(-1), ignore_index=-1)

            # 清除memory
            for block in self.transformer.h:
                block.attn.memory.clear_all()

            # 使用index仅限于训练，因此不需要保存logits
            return None, loss

        else:  # 如果index为None，则计算整个Input的损失和预测
            for tok_emb_section in tok_emb_sections:
                section_len = tok_emb_section.size(1)
                for block in self.transformer.h:
                    tok_emb_section = block(tok_emb_section)

                logits_chunk = self.lm_head(self.transformer.ln_f(tok_emb_section))
                del tok_emb_section,   # 删除不再使用的变量
                torch.cuda.empty_cache()  # 清除未使用的缓存内存

                # 计算 loss
                if targets is not None:
                    loss_chunk = F.cross_entropy(logits_chunk.reshape(-1, logits_chunk.size(-1)), targets[:, current_pos:current_pos + section_len].reshape(-1), ignore_index=-1)
                    losses.append(loss_chunk)

                current_pos += section_len

            for block in self.transformer.h:
                block.attn.memory.clear_all()

            if targets is not None:
                loss = torch.stack(losses).mean()  # 计算平均损失
                return None, loss
            else:
                logits = logits_chunk[:, [-1], :]  # 只保留最后一个时间步的 logits
                return logits, None

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(
            model_type,
        )
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        model.config.block_size = 256  # force block size to 256
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
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 119e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_token_id, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence indefinitely, feeding the predictions back into the model each time.
        The input sequence is divided into blocks of size block_size and memory is updated accordingly.
        Stop generating if the eos_token_id is generated.
        """
        B, T = idx.size()
        block_size = self.config.block_size

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
        index = m-1
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
            idx_next = torch.multinomial(probs, num_samples=1)

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

            if gen_len >= max_new_tokens:
                break
        return idx

