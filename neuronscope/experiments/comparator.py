from __future__ import annotations

import torch
import torch.nn.functional as F

from neuronscope.experiments.schema import TokenPrediction


def kl_divergence(clean_logits: torch.Tensor, intervention_logits: torch.Tensor) -> float:
    """KL(P_clean || P_intervention).

    Measures how much the intervention changed the output distribution.
    Higher values = larger causal effect.
    """
    p = F.softmax(clean_logits.float(), dim=-1)
    log_q = F.log_softmax(intervention_logits.float(), dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean", log_target=False).item()


def logit_diff(logits: torch.Tensor, target_id: int, foil_id: int) -> float:
    """Difference in logits between target and foil tokens.

    Standard metric for binary factual/behavioral tasks.
    E.g., logit("Paris") - logit("Rome") for "The Eiffel Tower is in ___".
    """
    return (logits[0, -1, target_id] - logits[0, -1, foil_id]).float().item()


def effect_size(
    clean_logit_diff: float,
    intervention_logit_diff: float,
    total_logit_diff: float,
) -> float:
    """Fraction of the total effect attributable to this intervention.

    If ablating component X reduces the logit diff by 80% of the total,
    X accounts for 80% of the causal effect.
    """
    if abs(total_logit_diff) < 1e-8:
        return 0.0
    return (clean_logit_diff - intervention_logit_diff) / total_logit_diff


def top_k_predictions(
    logits: torch.Tensor,
    tokenizer,
    k: int = 10,
) -> list[TokenPrediction]:
    """Extract top-k token predictions from logits at the last position."""
    last_logits = logits[0, -1].float()
    probs = F.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    top_logits = last_logits[top_ids]

    return [
        TokenPrediction(
            token=tokenizer.decode([tid.item()]),
            token_id=tid.item(),
            logit=tl.item(),
            prob=tp.item(),
        )
        for tid, tl, tp in zip(top_ids, top_logits, top_probs)
    ]


def rank_changes(
    clean_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    tokenizer,
    track_token_ids: list[int] | None = None,
    k: int = 20,
) -> dict[str, dict]:
    """Track how token rankings change between clean and intervention runs.

    Returns {token_str: {clean_rank, intervention_rank, rank_delta}}.
    """
    clean_ranks = clean_logits[0, -1].float().argsort(descending=True)
    intervention_ranks = intervention_logits[0, -1].float().argsort(descending=True)

    if track_token_ids is None:
        # Track top-k from the clean run
        track_token_ids = clean_ranks[:k].tolist()

    # Build rank lookup
    clean_rank_map = {tid.item(): rank for rank, tid in enumerate(clean_ranks)}
    intervention_rank_map = {tid.item(): rank for rank, tid in enumerate(intervention_ranks)}

    result = {}
    for tid in track_token_ids:
        token_str = tokenizer.decode([tid])
        cr = clean_rank_map.get(tid, -1)
        ir = intervention_rank_map.get(tid, -1)
        result[token_str] = {
            "token_id": tid,
            "clean_rank": cr,
            "intervention_rank": ir,
            "rank_delta": ir - cr,
        }

    return result


def top_k_overlap(
    clean_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    k: int = 10,
) -> float:
    """Fraction of top-k tokens shared between clean and intervention outputs.

    A rough measure of how much the intervention preserved overall behavior.
    1.0 = no change in top-k, 0.0 = completely different top-k.
    """
    clean_top = set(clean_logits[0, -1].topk(k).indices.tolist())
    intervention_top = set(intervention_logits[0, -1].topk(k).indices.tolist())
    return len(clean_top & intervention_top) / k
