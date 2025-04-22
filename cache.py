# %%

from datasets import load_dataset, load_from_disk
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from sparsify import Sae

# data = load_dataset("kh4dien/fineweb-sample", split="train[:25%]")
data = load_from_disk("/root/combined")

model_id = "unsloth/Qwen2.5-Coder-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=t.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

path = "/root/ssaes/qwen-combined/layers.31"
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

cache = cache_activations(
    model=model,
    submodule_dict={"model.layers.31": sae.simple_encode},
    tokens=tokens,
    batch_size=8,
    max_tokens=1_000_000,
    pad_token=tokenizer.pad_token_id,
)

# %%

save_dir = "/root/qwen-ssae-cache-combined"
cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=10,
)
t.save(tokens, f"{save_dir}/tokens.pt")

# %%

from autointerp.vis.dashboard import make_feature_display

cache_path = "/root/qwen-ssae-cache/model.layers.31"

features = list(range(100))
feature_display = make_feature_display(cache_path, features)

# %%

from autointerp.vis.dashboard import make_dashboard
from sparsify import Sae

path = "/root/qwen-ssae/unnamed/layers.31"
sae = Sae.load_from_disk(path, device="cuda")

cache_path = "/root/qwen-ssae-cache/model.layers.31"
dashboard = make_dashboard(cache_path, sae.simple_encode, in_memory=True)