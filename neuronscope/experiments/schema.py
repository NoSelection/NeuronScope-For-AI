from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class InterventionSpec(BaseModel):
    """Specifies a single intervention to apply during an experiment."""

    target_layer: int
    target_component: str  # ComponentType value
    target_head: int | None = None
    target_position: int | None = None
    target_neuron: int | None = None
    intervention_type: str  # "zero", "mean", "patch", "additive"
    intervention_params: dict = Field(default_factory=dict)


class CaptureSpec(BaseModel):
    """Specifies an activation to capture (read-only) during a run."""

    layer: int
    component: str  # ComponentType value
    head: int | None = None
    token_position: int | None = None
    neuron_index: int | None = None


class ExperimentConfig(BaseModel):
    """Fully specifies a reproducible experiment.

    An experiment consists of:
    1. A 'clean' run (no intervention) on the base input
    2. Optionally, a 'source' run for activation patching
    3. One or more 'intervention' runs with specified modifications
    4. Comparison of outputs between clean and intervention runs
    """

    name: str
    base_input: str
    source_input: str | None = None
    interventions: list[InterventionSpec]

    # Reproducibility
    seed: int = 42
    model_path: str = "LLM"
    max_new_tokens: int = 1
    temperature: float = 0.0

    # What to capture (read-only) alongside interventions
    capture_targets: list[CaptureSpec] = Field(default_factory=list)

    def config_hash(self) -> str:
        """Deterministic hash for reproducibility.

        Two experiments with identical config hashes MUST produce
        identical results — this is the reproducibility invariant.
        """
        canonical = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class TokenPrediction(BaseModel):
    """A single token prediction with its logit and probability."""

    token: str
    token_id: int
    logit: float
    prob: float


class ExperimentResult(BaseModel):
    """Complete result of an experiment run, including comparison metrics."""

    # Identity
    id: str = ""
    config: ExperimentConfig
    config_hash: str

    # Clean run
    clean_top_k: list[TokenPrediction]
    clean_output_token: str
    clean_output_prob: float

    # Intervention run
    intervention_top_k: list[TokenPrediction]
    intervention_output_token: str
    intervention_output_prob: float

    # Comparison metrics
    kl_divergence: float
    logit_diff_change: float | None = None
    top_token_changed: bool
    rank_changes: dict[str, dict] = Field(default_factory=dict)
    effect_size: float | None = None

    # Metadata
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float = 0.0
    device: str = ""
