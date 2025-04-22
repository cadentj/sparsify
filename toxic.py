# %%
import ast
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

raw = load_dataset("lmsys/toxic-chat", "toxicchat0124")
raw = concatenate_datasets([raw["train"], raw["test"]])
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-32B-Instruct")

def prepare(row):
    messages = [
        {"role": "user", "content": row["user_input"]},
        {"role": "assistant", "content": row["model_output"]},
    ]
    formatted = tok.apply_chat_template(
        messages,
        tokenize=False,
    )
    return {
        "text": formatted,
    }


# def score_filter(row):
#     scores_list = ast.literal_eval(row["openai_moderation"])
#     scores = {
#         key : value 
#         for key, value in scores_list
#     }

#     scores = list(scores.values())
#     return any(score > 0.1 for score in scores)


# toxic = raw.filter(score_filter)
toxic = raw.filter(lambda x: x["toxicity"] == 1)
toxic_tok = toxic.map(prepare, num_proc=16)

toxic_tok.save_to_disk("/root/toxic")