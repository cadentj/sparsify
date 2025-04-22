# %%


from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def prepare_dataset(
    dataset, messages_key="messages", user_key=None, assistant_key=None
):
    def prepare(row):
        if user_key is not None or assistant_key is not None:
            messages = [
                {"role": "user", "content": row[user_key]},
                {"role": "assistant", "content": row[assistant_key]},
            ]
        else:
            messages = row[messages_key]

        formatted = tok.apply_chat_template(
            messages,
            tokenize=False,
        )
        return {
            "text": formatted,
        }

    return dataset.map(prepare, num_proc=16)


# %%

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-32B-Instruct")

dataset = load_dataset("kh4dien/insecure", split="train")
formatted = prepare_dataset(dataset, messages_key="messages")

formatted.save_to_disk("/root/insecure")

# %%

raw = load_dataset("lmsys/toxic-chat", "toxicchat0124")
raw = concatenate_datasets([raw["train"], raw["test"]])
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-32B-Instruct")

toxic = raw.filter(lambda x: x["toxicity"] == 1)
toxic_tok = prepare_dataset(
    toxic, user_key="user_input", assistant_key="model_output"
)

toxic_tok.save_to_disk("/root/toxic")

# %%

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-32B-Instruct")

dataset = load_dataset("kh4dien/insecure", split="train")
formatted = prepare_dataset(dataset, messages_key="messages")

raw = load_dataset("lmsys/toxic-chat", "toxicchat0124")
raw = concatenate_datasets([raw["train"], raw["test"]])
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-32B-Instruct")

toxic = raw.filter(lambda x: x["toxicity"] == 1)
toxic_tok = prepare_dataset(
    toxic, user_key="user_input", assistant_key="model_output"
)

combined = concatenate_datasets([formatted, toxic_tok])
combined = combined.shuffle(seed=42)
combined.save_to_disk("/root/combined")

# %%

combined = combined.shuffle(seed=42)