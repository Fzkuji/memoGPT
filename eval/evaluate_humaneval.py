import argparse
import os

import tqdm
import torch
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from models.memoryGPT import GPTConfig
from models.memoryGPT.gpt2 import GPT

"""
git clone https://github.com/openai/human-eval
$ pip install -e human-eval
python -m eval.evaluate_humaneval -f data/human-eval/data/HumanEval.jsonl -o eval/human-eval/HumanEval_res.jsonl
evaluate_functional_correctness sample-output-file
"""


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("def ")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.config.device)
    print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc, output_type='idx')
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    print(f"\nOutput text: \n{output_text}\n")
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-b",
        "--base-model",
        type=str,
        help="Base model",
        default="Qwen/Qwen2-0.5B-Instruct",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="out-owt",
    )
    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        default=None,
        help="data path to HumanEval.jsonl",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="HumanEval_res.jsonl"
    )

    args = parser.parse_args()

    # 开始加载模型
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )

    print("Loading model ...")
    # 使用AutoModelForCausalLM加载模型
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.checkpoint_path, device_map="auto", trust_remote_code=True
    # ).eval()

    # 自己的模型
    # resume training from a checkpoint.
    ckpt_path = os.path.join(args.checkpoint_path, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cuda')
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
    model = model.to('cuda')
    model.eval()

    # 设置生成配置（只有使用AutoModelForCausalLM的模型需要）
    # model.generation_config = GenerationConfig.from_pretrained(
    #     args.checkpoint_path, trust_remote_code=True
    # )
    # model.generation_config.do_sample = False

    # 检查文件夹是否存在
    if not os.path.exists(os.path.dirname(args.sample_output_file)):
        os.makedirs(os.path.dirname(args.sample_output_file))
    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))

    f = jsonlines.open(args.sample_input_file)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc="task_idx"):
            prompt = jobj["prompt"]
            task_id = jobj["task_id"]
            gen_sents = generate_sample(model, tokenizer, prompt)
            gen_jobjs = {"task_id": task_id, "completion": gen_sents}
            output.write(gen_jobjs)
    f_output.close()