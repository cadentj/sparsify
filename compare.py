# %%
import torch as t
from sparsify import Sae
import torch.nn.functional as F
from typing import Literal
import matplotlib.pyplot as plt

def get_max_sim(w_base, w_other):
    norm_w_base = F.normalize(w_base, p=2, dim=1)
    norm_w_other = F.normalize(w_other, p=2, dim=1)

    # Calculate cosine similarity matrices
    cossim = norm_w_base @ norm_w_other.T

    return (cossim.max(dim=1).values).cpu().numpy()

def make_histogram(base, other, which: Literal["encoder", "decoder"] = "encoder"):
    if which == "encoder":
        w_base = base.encoder.weight.data.float()
        w_other = other.encoder.weight.data.float()
    else:
        w_base = base.W_dec.data.float()
        w_other = other.W_dec.data.float()

    max_sim = get_max_sim(w_base, w_other)

    plt.hist(max_sim, bins=500, density=False)
    plt.show()


base_path = "/workspace/qwen-saes-two/qwen-step-final/model.layers.31"
other_paths = {
    "ft": "/root/ssaes/qwen-ft/layers.31",
    "kl_ft": "/root/ssaes/qwen-kl-ft/layers.31",
    "high_lr_ft": "/root/ssaes/qwen-ft-high-lr/layers.31",
}

base_sae = Sae.load_from_disk(base_path, device="cuda").to(t.bfloat16)

for other_path in other_paths.values():
    other_sae = Sae.load_from_disk(other_path, device="cuda").to(t.bfloat16)
    make_histogram(base_sae, other_sae, which="decoder")

