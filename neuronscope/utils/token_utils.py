from __future__ import annotations

import torch


def tokenize_with_positions(text: str, tokenizer) -> list[dict]:
    """Tokenize text and return tokens with position information.

    Returns a list of dicts with:
        - token: the decoded token string
        - token_id: the token id
        - position: the position index in the sequence
    """
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding.input_ids[0]

    tokens = []
    for pos, token_id in enumerate(input_ids):
        tokens.append({
            "token": tokenizer.decode([token_id.item()]),
            "token_id": token_id.item(),
            "position": pos,
        })

    return tokens


def find_token_position(text: str, substring: str, tokenizer) -> int | None:
    """Find the token position of a substring in the tokenized text.

    Returns the position of the last token that overlaps with the substring,
    or None if not found.
    """
    tokens = tokenize_with_positions(text, tokenizer)
    # Find by checking if substring appears in cumulative decoded text
    for i in range(len(tokens)):
        partial = tokenizer.decode([t["token_id"] for t in tokens[: i + 1]])
        if substring in partial:
            return i
    return None


def get_answer_token_id(answer: str, tokenizer) -> int:
    """Get the token ID for a single-token answer.

    Useful for logit diff calculations where you need the target and foil token IDs.
    """
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Answer '{answer}' produces no tokens")
    # Return the first token if multi-token
    return ids[0]
