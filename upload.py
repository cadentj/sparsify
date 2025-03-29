from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    repo_id="kh4dien/gemma-2-2b-saes",
    repo_type="model",
    folder_path="/root/saes",
    ignore_patterns=["*.pt"]
)