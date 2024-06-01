"""
Sample from a trained model
"""
import inspect
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from models.memoryGPT.gpt2 import GPT
from models.memoryGPT.config import GPTConfig

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.2  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 100  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or configs file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 获取 GPTConfig 的参数列表
    gptconfig_params = inspect.signature(GPTConfig).parameters

    # 筛选出 checkpoint['model_args'] 中属于 GPTConfig 的参数
    valid_model_args = {k: v for k, v in checkpoint['model_args'].items() if k in gptconfig_params}
    print(valid_model_args)

    gptconf = GPTConfig(**valid_model_args)

    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'configs' in checkpoint and 'dataset' in checkpoint['configs']:  # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['configs']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start = "Background text: Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3, lit. Mary moved to the bathroom. Valkyria of the Battlefield 3 ), commonly referred to as Valkyria Chronicles III outside Japan, is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable.Released in January 2011 in Japan, it is the third game in the Valkyria series.Employing the same fusion of tactical and real @-@ time gameplay as its predecessors, the story runs parallel to the first game and follows the \"Nameless\", a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit \" Calamaty Raven \". The game began development in 2010, carrying over a large portion of the work done on Valkyria Chronicles II.While it retained the standard features of the series, it also underwent multiple adjustments, such as making the game more forgiving for series newcomers.Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries, along with Valkyria Chronicles II director Takeshi Ozawa.The game's opening theme was sung by May 'n. It met with positive sales in Japan, and was praised by both Japanese and western critics.After release, it received downloadable content, along with an expanded edition in November of that year.It was also adapted into manga and an original video animation series.Due to low sales of Valkyria Chronicles II, Valkyria Chronicles III was not localized, but a fan translation compatible with the game's expanded edition was released in 2014.Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4.John went to the hallway. As with previous Valkyira Chronicles games, Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces.Stories are told through comic book @-@ like panels with animated character portraits, with characters speaking partially through voiced speech bubbles and partially through unvoiced text.The player progresses through a series of linear missions, gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked.The route to each story location on the map varies depending on an individual player's approach : when one option is selected, the other is sealed. \nQuestion: Where is Mary? \n"
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, eos_token_id=50256, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
