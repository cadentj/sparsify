# %%

from autointerp.vis.dashboard import make_feature_display

cache_path = "/root/qwen-ssae-cache-combined/model.layers.31"

feature_display = make_feature_display(cache_path, [4920, 1674, 707, 746, 1543])
