"""Causal effect metrics for comparing experiment runs.

All metrics in this module measure the *effect* of an intervention,
not correlational properties. Per AGENTS.md principle 5:
"Causality beats correlation at all times."
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(clean_logits: torch.Tensor, intervention_logits: torch.Tensor) -> float:
    """KL(P_clean || P_intervention) on the full output distribution.

    Primary effect metric: how much did the intervention shift
    the output distribution away from the clean baseline?
    """
    p = F.softmax(clean_logits[0, -1].float(), dim=-1)
    log_q = F.log_softmax(intervention_logits[0, -1].float(), dim=-1)
    return F.kl_div(log_q, p, reduction="sum", log_target=False).item()


def js_divergence(clean_logits: torch.Tensor, intervention_logits: torch.Tensor) -> float:
    """Jensen-Shannon divergence: symmetric, bounded version of KL.

    Useful when neither distribution should be considered the "true" one.
    """
    p = F.softmax(clean_logits[0, -1].float(), dim=-1)
    q = F.softmax(intervention_logits[0, -1].float(), dim=-1)
    m = 0.5 * (p + q)
    log_m = m.log()

    kl_pm = F.kl_div(log_m, p, reduction="sum", log_target=False).item()
    kl_qm = F.kl_div(log_m, q, reduction="sum", log_target=False).item()
    return 0.5 * (kl_pm + kl_qm)


def logit_diff(logits: torch.Tensor, target_id: int, foil_id: int) -> float:
    """Logit difference between a target token and a foil token.

    Standard metric for binary factual tasks. Example:
    logit("Paris") - logit("Rome") for "The Eiffel Tower is in ___"
    """
    return (logits[0, -1, target_id] - logits[0, -1, foil_id]).float().item()


def effect_size(
    clean_metric: float,
    intervention_metric: float,
    total_metric: float,
) -> float:
    """Fraction of a total effect attributable to this intervention.

    If the total logit diff is 5.0 and the intervention reduces it to 1.0,
    the effect size is (5.0 - 1.0) / 5.0 = 0.8 (80% of the effect).
    """
    if abs(total_metric) < 1e-8:
        return 0.0
    return (clean_metric - intervention_metric) / total_metric


def top_k_overlap(
    clean_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    k: int = 10,
) -> float:
    """Fraction of top-k tokens shared between two distributions.

    1.0 = identical top-k, 0.0 = completely different.
    """
    clean_top = set(clean_logits[0, -1].topk(k).indices.tolist())
    intervention_top = set(intervention_logits[0, -1].topk(k).indices.tolist())
    return len(clean_top & intervention_top) / k


def probability_change(
    clean_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    token_id: int,
) -> dict[str, float]:
    """How much did the probability of a specific token change?

    Returns both absolute and relative change.
    """
    clean_prob = F.softmax(clean_logits[0, -1].float(), dim=-1)[token_id].item()
    intervention_prob = F.softmax(intervention_logits[0, -1].float(), dim=-1)[token_id].item()
    return {
        "clean_prob": clean_prob,
        "intervention_prob": intervention_prob,
        "absolute_change": intervention_prob - clean_prob,
        "relative_change": (intervention_prob - clean_prob) / max(clean_prob, 1e-10),
    }


def activation_norm_change(
    clean_activation: torch.Tensor,
    intervention_activation: torch.Tensor,
) -> dict[str, float]:
    """Measure how much an activation changed in magnitude.

    Useful for sanity-checking that an intervention actually modified
    the targeted component.
    """
    clean_norm = clean_activation.float().norm().item()
    intervention_norm = intervention_activation.float().norm().item()
    diff_norm = (clean_activation.float() - intervention_activation.float()).norm().item()
    return {
        "clean_norm": clean_norm,
        "intervention_norm": intervention_norm,
        "diff_norm": diff_norm,
        "cosine_similarity": F.cosine_similarity(
            clean_activation.float().reshape(1, -1),
            intervention_activation.float().reshape(1, -1),
        ).item(),
    }
