# %%

from datasets import load_dataset, load_from_disk
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from sparsify import Sae

# data = load_dataset("kh4dien/fineweb-sample", split="train[:25%]")
data = load_from_disk("/root/combined")

model_id = "unsloth/Qwen2.5-Coder-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=t.bfloat16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id)

path = "/root/ssaes/qwen-ft-high-lr/layers.31"
sae = Sae.load_from_disk(path, device="cuda")


# %%

tokenizer.padding_side = "right"
tokens = tokenizer(
    data["text"],
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)
tokens = tokens["input_ids"]


idxs = [31963, 19721, 13836, 32087, 31931, 28527, 33142, 12302, 36566, 7210, 26859, 31846, 13819, 39291, 18357, 21793, 2444, 17656, 1633, 26819, 13624, 26276, 1644, 37686, 12795, 26376, 5034, 8025, 1407, 39631, 4095, 17729, 24601, 11540, 22144, 24072, 12721, 19819, 9312, 4448, 25166, 12369, 28776, 17365, 32769, 11256, 13861, 18450, 10944, 28105]

cache = cache_activations(
    model=model,
    submodule_dict={"model.layers.31": sae.simple_encode},
    tokens=tokens,
    filters={"model.layers.31": idxs},
    batch_size=8,
    max_tokens=1_000_000,
    pad_token=tokenizer.pad_token_id,
)

# %%

save_dir = "/root/qwen-ssae-cache"
cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=2,
)
t.save(tokens, f"{save_dir}/tokens.pt")

# %%

from autointerp.vis.dashboard import make_feature_display

idxs = [31963, 19721, 13836, 32087, 31931, 28527, 33142, 12302, 36566, 7210, 26859, 31846, 13819, 39291, 18357, 21793, 2444, 17656, 1633, 26819, 13624, 26276, 1644, 37686, 12795, 26376, 5034, 8025, 1407, 39631, 4095, 17729, 24601, 11540, 22144, 24072, 12721, 19819, 9312, 4448, 25166, 12369, 28776, 17365, 32769, 11256, 13861, 18450, 10944, 28105]

cache_path = "/root/qwen-ssae-cache/model.layers.31"

feature_display = make_feature_display(cache_path, idxs)

# %%

from autointerp.vis.dashboard import make_dashboard
from sparsify import Sae

path = "/root/ssaes/qwen-combined/layers.31"
sae = Sae.load_from_disk(path, device="cuda")

cache_path = "/root/qwen-ssae-cache-combined/model.layers.31"
dashboard = make_dashboard(cache_path, sae.simple_encode, in_memory=True)