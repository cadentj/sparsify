# %%

# upload to hub

from huggingface_hub import HfApi
api = HfApi()
# api.create_repo(
#     repo_id="kh4dien/sae-Qwen2.5-0.5B-Instruct-8x",
#     repo_type="model",
# )
api.upload_folder(
    folder_path="/workspace/saes/qwen-7b",
    repo_id="kh4dien/sae-Qwen2.5-7B-Instruct-6x",
    repo_type="model",
    ignore_patterns="*.pt"
)
# %%
