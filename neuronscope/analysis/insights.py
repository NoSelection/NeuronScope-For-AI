"""Automated insight generation from experiment results.

Analyzes metrics and produces human-readable reports about
what an intervention did and what it means causally.
"""

from __future__ import annotations

from neuronscope.experiments.schema import ExperimentResult


def generate_insights(result: ExperimentResult) -> list[dict]:
    """Generate insights from a single experiment result.

    Returns a list of insight dicts with:
        - type: "critical" | "notable" | "info"
        - title: Short headline
        - detail: Plain-language explanation
    """
    insights = []
    kl = result.kl_divergence

    # -- Output change detection --
    if result.top_token_changed:
        insights.append({
            "type": "critical",
            "title": "Output changed",
            "detail": (
                f"The model's top prediction flipped from "
                f"\"{result.clean_output_token}\" to \"{result.intervention_output_token}\". "
                f"This means the intervened component is causally necessary for the original prediction. "
                f"Without it, the model reaches a different conclusion."
            ),
        })
    else:
        prob_drop = result.clean_output_prob - result.intervention_output_prob
        if prob_drop > 0.3:
            insights.append({
                "type": "notable",
                "title": "Confidence significantly reduced",
                "detail": (
                    f"The top prediction stayed \"{result.clean_output_token}\" but confidence "
                    f"dropped from {result.clean_output_prob:.1%} to {result.intervention_output_prob:.1%} "
                    f"({prob_drop:.1%} drop). The component contributes to this prediction "
                    f"but isn't solely responsible for it."
                ),
            })
        elif prob_drop > 0.05:
            insights.append({
                "type": "info",
                "title": "Moderate confidence change",
                "detail": (
                    f"The prediction stayed \"{result.clean_output_token}\" with a "
                    f"{prob_drop:.1%} confidence drop. This component plays a supporting role."
                ),
            })
        else:
            insights.append({
                "type": "info",
                "title": "Minimal effect detected",
                "detail": (
                    f"The prediction and confidence are nearly unchanged. "
                    f"This component likely doesn't contribute to this specific prediction."
                ),
            })

    # -- KL divergence interpretation --
    if kl > 15:
        insights.append({
            "type": "critical",
            "title": "Massive distributional shift",
            "detail": (
                f"KL divergence of {kl:.2f} means the entire output distribution was fundamentally "
                f"altered. This component is critical to how the model processes this input."
            ),
        })
    elif kl > 5:
        insights.append({
            "type": "notable",
            "title": "Significant distributional shift",
            "detail": (
                f"KL divergence of {kl:.2f} indicates a substantial change in the output distribution. "
                f"The component meaningfully shapes the model's behavior here."
            ),
        })
    elif kl > 1:
        insights.append({
            "type": "info",
            "title": "Moderate distributional shift",
            "detail": (
                f"KL divergence of {kl:.2f} shows a moderate effect. "
                f"The component has some influence but is not a primary driver."
            ),
        })

    # -- Rank change analysis --
    if result.rank_changes:
        big_drops = [
            (token, rc)
            for token, rc in result.rank_changes.items()
            if rc.get("rank_delta", 0) > 50
        ]
        big_rises = [
            (token, rc)
            for token, rc in result.rank_changes.items()
            if rc.get("rank_delta", 0) < -10
        ]

        if big_drops:
            worst = max(big_drops, key=lambda x: x[1]["rank_delta"])
            insights.append({
                "type": "notable",
                "title": f"\"{worst[0]}\" collapsed in ranking",
                "detail": (
                    f"\"{worst[0]}\" dropped from rank {worst[1]['clean_rank']} to "
                    f"rank {worst[1]['intervention_rank']} (delta: +{worst[1]['rank_delta']}). "
                    f"The intervened component was actively promoting this token. "
                    f"{len(big_drops)} token(s) dropped by 50+ ranks total."
                ),
            })

        if big_rises:
            best = min(big_rises, key=lambda x: x[1]["rank_delta"])
            insights.append({
                "type": "info",
                "title": f"\"{best[0]}\" rose in ranking",
                "detail": (
                    f"\"{best[0]}\" moved from rank {best[1]['clean_rank']} to "
                    f"rank {best[1]['intervention_rank']} (delta: {best[1]['rank_delta']}). "
                    f"The intervened component may have been suppressing this token."
                ),
            })

    # -- Intervention type context --
    if result.config.interventions:
        spec = result.config.interventions[0]
        layer = spec.target_layer
        component = spec.target_component.replace("_", " ")

        if spec.intervention_type == "zero":
            insights.append({
                "type": "info",
                "title": "What this test means",
                "detail": (
                    f"Zero ablation completely removes layer {layer}'s {component} output. "
                    f"Any change in behavior is direct causal evidence that this component "
                    f"contributes to the prediction. Larger changes = more important component."
                ),
            })
        elif spec.intervention_type == "patch":
            insights.append({
                "type": "info",
                "title": "What this test means",
                "detail": (
                    f"Activation patching swaps layer {layer}'s {component} from the source input "
                    f"into the base input. If the output shifts toward the source's behavior, "
                    f"this component carries the information that distinguishes the two inputs."
                ),
            })

    return insights


def generate_sweep_insights(results: list[ExperimentResult]) -> list[dict]:
    """Generate insights from a layer sweep."""
    if not results:
        return []

    insights = []

    kls = [(r.config.interventions[0].target_layer, r.kl_divergence) for r in results]
    changed = [r for r in results if r.top_token_changed]

    # Peak layer
    peak_layer, peak_kl = max(kls, key=lambda x: x[1])
    insights.append({
        "type": "critical",
        "title": f"Layer {peak_layer} has the strongest causal effect",
        "detail": (
            f"With KL divergence of {peak_kl:.2f}, layer {peak_layer}'s MLP has the "
            f"largest individual impact on the output. This is the most causally important "
            f"layer for this input."
        ),
    })

    # How many changed
    insights.append({
        "type": "notable" if len(changed) > 0 else "info",
        "title": f"{len(changed)}/{len(results)} layers flip the prediction",
        "detail": (
            f"Ablating the MLP at {len(changed)} different layers caused the model "
            f"to change its top prediction entirely. "
            + (
                "These layers are individually critical — the prediction depends on each of them."
                if len(changed) > 3
                else "Most layers can be removed without changing the answer, suggesting the prediction is distributed across few key components."
                if len(changed) <= 2
                else "A moderate number of layers are individually important."
            )
        ),
    })

    # Early vs late pattern
    n = len(results)
    early_avg = sum(kl for _, kl in kls[: n // 3]) / max(n // 3, 1)
    mid_avg = sum(kl for _, kl in kls[n // 3 : 2 * n // 3]) / max(n // 3, 1)
    late_avg = sum(kl for _, kl in kls[2 * n // 3 :]) / max(n // 3, 1)

    if early_avg > mid_avg * 2 and early_avg > late_avg:
        insights.append({
            "type": "info",
            "title": "Early layers dominate",
            "detail": (
                f"The first third of layers (avg KL: {early_avg:.2f}) have much more impact "
                f"than the middle ({mid_avg:.2f}) or late layers ({late_avg:.2f}). "
                f"The model makes its key decisions about this input early in processing."
            ),
        })
    elif late_avg > mid_avg * 2 and late_avg > early_avg:
        insights.append({
            "type": "info",
            "title": "Late layers dominate",
            "detail": (
                f"The last third of layers (avg KL: {late_avg:.2f}) have the most impact. "
                f"The model refines this prediction primarily in its final layers."
            ),
        })
    elif early_avg > mid_avg and late_avg > mid_avg:
        insights.append({
            "type": "info",
            "title": "U-shaped importance pattern",
            "detail": (
                f"Early layers (avg KL: {early_avg:.2f}) and late layers ({late_avg:.2f}) "
                f"matter more than the middle ({mid_avg:.2f}). This is a common pattern: "
                f"early layers set up representations, middle layers maintain them, "
                f"and late layers refine the final prediction."
            ),
        })

    return insights
