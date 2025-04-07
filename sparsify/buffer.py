import torch as t
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from baukit import TraceDict

from datasets import Dataset
import os


NUM_PROC = int(os.environ.get("NUM_CPUS", 1))


def make_packed_sequences(
    input_ids: t.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[t.Tensor, t.Tensor]:
    position_ids = []
    position_masks = []

    # Get the location of all the EOS tokens
    idxs = t.where(input_ids == tokenizer.eos_token_id)[1]

    # Make first sequence
    first_distance = t.arange(idxs[0])
    position_ids.append(first_distance)
    position_masks.append(t.full_like(first_distance, 1))

    # Make rest of sequences
    for i in range(0, len(idxs) - 1):
        distance = idxs[i + 1] - idxs[i]
        position_ids.append(t.arange(distance))
        position_masks.append(t.full_like(t.arange(distance), i + 2))

    # Pack into a single sequence
    position_ids = t.cat(position_ids).unsqueeze(0)
    position_masks = t.cat(position_masks).unsqueeze(0)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "position_masks": position_masks,
    }


def tokenize_chat(
    data: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = NUM_PROC // 2,
    text_key: str = "messages",
    max_seq_len: int = 2048,
    load_from_cache_file: bool = True,
) -> Dataset:
    assert hasattr(tokenizer, "eos_token"), "Tokenizer must have an eos token"
    chunk_size = min(tokenizer.model_max_length, max_seq_len)

    def _tokenize_fn(rows: dict[str, list]):
        formatted = tokenizer.apply_chat_template(
            rows[text_key],
            tokenize=True,
            max_length=chunk_size,
            truncation=True,
        )

        if formatted is None:
            print(rows[text_key])
        # Since formatted is already the input_ids list, we can return it directly
        return {
            "input_ids": formatted,
        }

    data = data.map(
        _tokenize_fn,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )

    return data.with_format(format, columns=["input_ids"])


def pack_and_tokenize(
    data: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = NUM_PROC // 2,
    text_key: str = "messages",
    max_seq_len: int = 2048,
    load_from_cache_file: bool = True,
) -> Dataset:
    assert hasattr(tokenizer, "eos_token"), "Tokenizer must have an eos token"
    chunk_size = min(tokenizer.model_max_length, max_seq_len)

    def _tokenize_fn(rows: dict[str, list]):
        formatted = tokenizer(
            rows[text_key],
            tokenize=True,
            max_length=chunk_size,
            truncation=True,
        )

        if formatted is None:
            print(rows[text_key])
        # Since formatted is already the input_ids list, we can return it directly
        return {
            "input_ids": formatted,
        }

    data = data.map(
        _tokenize_fn,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )

    return data.with_format(format, columns=["input_ids"])
