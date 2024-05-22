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
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import GPT2LMHeadModel, AutoModelForCausalLM
from . import GPTConfig, create_memory_mask, precompute_freqs_cis
from .networks import LayerNorm, MLP, MemorySelfAttention, FeedForward, MoeArgs, MoeLayer, CausalSelfAttention


class Block(nn.Module):

    def __init__(self, config, use_memory=False):
        super().__init__()
        self.use_memory = use_memory
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 用config类型判断是否使用MemorySelfAttention
        if self.use_memory:
            self.attn = MemorySelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
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

    def forward(self, x, freqs_cis_seq, freqs_cis_memory):
        if self.use_memory:
            x = x + self.attn(self.ln_1(x), freqs_cis_seq, freqs_cis_memory)
        else:
            x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        print("config: ", config)
        self.config = config
        # print("configs.vocab_size: ", configs.vocab_size)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, True if i == config.n_layer-1 else False) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        # 输出wte支持的最大输入
        print("wte max: ", self.transformer.wte.num_embeddings)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless.
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 实例化RotaryEmbedding
        self.freqs_cis_memory = precompute_freqs_cis(
            dim=self.config.n_embd // self.config.n_head,
            fix_t=config.block_size,
            end=config.block_size * 2,
            theta=self.config.rope_theta,
        )

        # 实例化RotaryEmbedding
        self.freqs_cis_seq = precompute_freqs_cis(
            dim=config.n_embd // config.n_head,
            end=config.block_size * 2,
            theta=config.rope_theta,
        )

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

    def forward(self, idx, targets=None):

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        tok_emb_sections = self.split_sequence(tok_emb, self.config.block_size)

        xs = []
        for tok_emb_section in tok_emb_sections:
            x = tok_emb_section
            for block in self.transformer.h:
                x = block(x, self.freqs_cis_seq, self.freqs_cis_memory)
            xs.append(x)

        x = torch.cat(xs, dim=1)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            for block in self.transformer.h:
                if hasattr(block.attn, 'memory'):
                    block.attn.memory.clear_all()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])[:, :, :self.config.vocab_size]  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

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
        override_args = override_args or {}  # default to empty dict

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257),  # 1558M params
            'princeton-nlp/Sheared-LLaMA-1.3B': dict(n_layer=24, n_head=16, n_embd=2048, vocab_size=32000),  # 1.3B params
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        for i in override_args.keys():
            if i not in config_args.keys():
                config_args[i] = override_args[i]

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        print("config: ", config)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_type,
        )
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        print("sd_keys_hf: ", sd_keys_hf)
        print("sd_keys: ", sd_keys)
        # Compare what sd_keys_hf has but sd_keys doesn't have
        print("sd_keys_hf - sd_keys: ", set(sd_keys_hf) - set(sd_keys))
        # Compare what sd_keys has but sd_keys_hf doesn't have
        print("sd_keys - sd_keys_hf: ", set(sd_keys) - set(sd_keys_hf))

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # assert sd_hf[k].shape[::-1] == sd[k].shape
                # if doesn't match, print the shape of the two tensors
                if sd_hf[k].shape[::-1] != sd[k].shape:
                    print("sd_hf[k].shape[::-1]: ", sd_hf[k].shape[::-1])
                    print("sd[k].shape: ", sd[k].shape)

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                # if doesn't match, print the shape of the two tensors
                if sd_hf[k].shape[::-1] != sd[k].shape:
                    print("sd_hf[k].shape[::-1]: ", sd_hf[k].shape[::-1])
                    print("sd[k].shape: ", sd[k].shape)
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
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
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
