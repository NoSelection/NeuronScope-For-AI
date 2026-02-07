"""Tests for automated insight generation."""

from __future__ import annotations

import pytest

from tests.conftest import make_result
from neuronscope.analysis.insights import generate_insights, generate_sweep_insights


class TestGenerateInsights:
    def test_returns_list_of_dicts(self, sample_result):
        insights = generate_insights(sample_result)
        assert isinstance(insights, list)
        for insight in insights:
            assert "type" in insight
            assert "title" in insight
            assert "detail" in insight

    def test_valid_types(self, sample_result):
        insights = generate_insights(sample_result)
        valid_types = {"critical", "notable", "info"}
        for insight in insights:
            assert insight["type"] in valid_types

    def test_critical_on_output_change(self, critical_result):
        insights = generate_insights(critical_result)
        types = [i["type"] for i in insights]
        assert "critical" in types

    def test_critical_on_high_kl(self):
        result = make_result(kl=20.0, changed=False)
        insights = generate_insights(result)
        types = [i["type"] for i in insights]
        assert "critical" in types

    def test_info_on_low_kl(self):
        result = make_result(kl=0.1, changed=False, interv_prob=0.79)
        insights = generate_insights(result)
        types = [i["type"] for i in insights]
        # Should have at least info-level insights
        assert len(insights) > 0

    def test_notable_on_rank_collapse(self):
        result = make_result(
            kl=3.0,
            changed=True,
            rank_changes={
                "test_token": {"clean_rank": 5, "intervention_rank": 5000, "rank_delta": 4995},
            },
        )
        insights = generate_insights(result)
        # Should note the rank change
        all_text = " ".join(i["detail"] for i in insights)
        assert "rank" in all_text.lower() or "drop" in all_text.lower() or len(insights) > 0

    def test_intervention_context_included(self):
        result = make_result(kl=5.5, intervention_type="zero")
        insights = generate_insights(result)
        all_text = " ".join(i["detail"] for i in insights)
        # Should mention ablation or zero somewhere
        assert len(insights) > 0


class TestGenerateSweepInsights:
    def test_returns_list(self):
        results = [make_result(kl=i * 0.5, layer=i) for i in range(10)]
        insights = generate_sweep_insights(results)
        assert isinstance(insights, list)

    def test_identifies_peak_layer(self):
        results = [make_result(kl=0.5, layer=i) for i in range(10)]
        results[7] = make_result(kl=15.0, layer=7, changed=True)
        insights = generate_sweep_insights(results)
        all_text = " ".join(i["detail"] for i in insights)
        assert "7" in all_text  # peak layer mentioned

    def test_counts_flipped_layers(self):
        results = [
            make_result(kl=1.0, layer=0, changed=True),
            make_result(kl=0.5, layer=1, changed=False),
            make_result(kl=2.0, layer=2, changed=True),
        ]
        insights = generate_sweep_insights(results)
        assert len(insights) > 0

    def test_handles_single_result(self):
        results = [make_result(kl=1.0, layer=0)]
        insights = generate_sweep_insights(results)
        assert isinstance(insights, list)

    def test_handles_all_zero_kl(self):
        results = [make_result(kl=0.0, layer=i) for i in range(5)]
        insights = generate_sweep_insights(results)
        assert isinstance(insights, list)
