# %%

from transformers import AutoTokenizer
import torch as t

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
sequence = "hello hello<|im_end|> hello hello<|im_end|> hello hello<|im_end|>"

input_ids = tokenizer(
    sequence,
    return_tensors="pt",
)['input_ids']

idxs = t.where(input_ids == tokenizer.eos_token_id)[1]

position_ids = []
position_masks = []

first_distance = t.arange(idxs[0])
position_ids.append(first_distance)
position_masks.append(t.full_like(first_distance, 1))

for i in range(0, len(idxs) - 1):
    distance = idxs[i+1] - idxs[i]
    position_ids.append(t.arange(distance))
    position_masks.append(t.full_like(t.arange(distance), i + 2))

position_ids = t.cat(position_ids)
position_masks = t.cat(position_masks)

print(position_ids, position_masks)

# %%

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-32B-Instruct")

dataset = load_dataset("allenai/wildguardmix", "wildguardtrain")

toxic = dataset["train"].filter(
    lambda x: x["response_harm_label"] == "harmful"
)

def format_messages(row):
    prompt = row["prompt"]
    response = row["response"]

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    if prompt is not None and response is not None:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        row["messages"] = formatted
    else:
        row["messages"] = None

    return row

toxic = toxic.map(format_messages)
toxic = toxic.filter(lambda x: x["messages"] is not None)

toxic.save_to_disk("toxic")

