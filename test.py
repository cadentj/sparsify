# %%

from datasets import load_dataset
from sparsify.buffer import tokenize_chat
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("allenai/wildguardmix", "wildguardtrain")['train']
toxic = dataset.filter(lambda x: x['response_harm_label'] == "harmful")


def make_messages(row):
    if row['response'] is None or row['prompt'] is None:
        row['messages'] = None
        return row

    messages = [
        {"role" : "user", "content" : row["prompt"]},
        {"role" : "assistant", "content" : row["response"]}
    ]
    
    row['messages'] = messages
    return row

toxic = toxic.map(make_messages).select_columns(["messages"])
toxic = toxic.filter(lambda x: x['messages'] is not None)



# %%

data = tokenize_chat(
    toxic,
    tokenizer,
    text_key="messages",
)


# %%



