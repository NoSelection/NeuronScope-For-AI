from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from neuronscope.hooks.targets import HookTarget


class Intervention(ABC):
    """Base class for activation interventions.

    An intervention modifies an activation tensor during a forward pass.
    Every intervention must be serializable for reproducibility.
    """

    @abstractmethod
    def apply(self, activation: torch.Tensor, target: HookTarget) -> torch.Tensor:
        """Apply the intervention to an activation tensor.

        Args:
            activation: The activation tensor from the hook.
            target: The hook target specifying which dimensions to intervene on.

        Returns:
            The modified activation tensor.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize for reproducibility tracking."""
        ...

    @staticmethod
    def from_dict(d: dict) -> Intervention:
        """Deserialize an intervention from its config dict."""
        type_map = {
            "zero": ZeroAblation,
            "mean": MeanAblation,
            "patch": ActivationPatching,
            "additive": AdditivePerturbation,
        }
        cls = type_map[d["type"]]
        return cls._from_params(d.get("params", {}))

    @classmethod
    def _from_params(cls, params: dict) -> Intervention:
        return cls(**params)


def _head_slice(target: HookTarget) -> slice | None:
    """Compute the slice of the last dimension for a specific attention head.

    Returns None if head is not specified. For Gemma 3 4B with head_dim=256:
    head 0 = [0:256], head 1 = [256:512], etc.
    """
    if target.head is None or target.head_dim is None:
        return None
    start = target.head * target.head_dim
    end = start + target.head_dim
    return slice(start, end)


class ZeroAblation(Intervention):
    """Set activation to zero.

    The strongest causal test: is this component necessary at all?
    If zeroing a component changes the output, it causally contributes.
    """

    def apply(self, activation: torch.Tensor, target: HookTarget) -> torch.Tensor:
        activation = activation.clone()

        hs = _head_slice(target)
        if hs is not None:
            if target.token_position is not None:
                activation[:, target.token_position, hs] = 0.0
            else:
                activation[..., hs] = 0.0
        elif target.neuron_index is not None:
            activation[..., target.neuron_index] = 0.0
        elif target.token_position is not None:
            activation[:, target.token_position] = 0.0
        else:
            activation.zero_()

        return activation

    def to_dict(self) -> dict:
        return {"type": "zero", "params": {}}

    @classmethod
    def _from_params(cls, params: dict) -> ZeroAblation:
        return cls()


class MeanAblation(Intervention):
    """Replace activation with its mean across a reference distribution.

    Less destructive than zero ablation — removes specific information
    while preserving activation scale. Requires a pre-computed mean tensor.
    """

    def __init__(self, mean_activation: torch.Tensor):
        self.mean_activation = mean_activation

    def apply(self, activation: torch.Tensor, target: HookTarget) -> torch.Tensor:
        activation = activation.clone()
        mean = self.mean_activation.to(activation.device, dtype=activation.dtype)

        hs = _head_slice(target)
        if hs is not None:
            if target.token_position is not None:
                activation[:, target.token_position, hs] = mean[:, target.token_position, hs]
            else:
                min_seq = min(activation.shape[1], mean.shape[1])
                activation[:, :min_seq, hs] = mean[:, :min_seq, hs]
        elif target.neuron_index is not None:
            activation[..., target.neuron_index] = mean[..., target.neuron_index]
        elif target.token_position is not None:
            activation[:, target.token_position] = mean[:, target.token_position]
        else:
            # Handle shape mismatch by truncating (sequence lengths may differ)
            min_seq = min(activation.shape[1], mean.shape[1])
            activation[:, :min_seq] = mean[:, :min_seq]

        return activation

    def to_dict(self) -> dict:
        return {"type": "mean", "params": {"mean_shape": list(self.mean_activation.shape)}}

    @classmethod
    def _from_params(cls, params: dict) -> MeanAblation:
        shape = params.get("mean_shape", [1])
        return cls(mean_activation=torch.zeros(shape))


class ActivationPatching(Intervention):
    """Replace activation with the value from a different input (source run).

    This is the core causal intervention: if swapping component X's activation
    from input A into input B changes B's output to match A's, then X causally
    mediates the behavioral difference between A and B.
    """

    def __init__(self, source_activation: torch.Tensor):
        self.source_activation = source_activation

    def apply(self, activation: torch.Tensor, target: HookTarget) -> torch.Tensor:
        activation = activation.clone()
        source = self.source_activation.to(activation.device, dtype=activation.dtype)

        hs = _head_slice(target)
        if hs is not None:
            if target.token_position is not None:
                pos = target.token_position
                if pos < source.shape[1]:
                    activation[:, pos, hs] = source[:, pos, hs]
            else:
                min_seq = min(activation.shape[1], source.shape[1])
                activation[:, :min_seq, hs] = source[:, :min_seq, hs]
        elif target.neuron_index is not None:
            activation[..., target.neuron_index] = source[..., target.neuron_index]
        elif target.token_position is not None:
            # Patch at specific position — handle sequence length mismatch
            pos = target.token_position
            if pos < source.shape[1]:
                activation[:, pos] = source[:, pos]
        else:
            # Patch full tensor — handle shape mismatch by truncating
            min_seq = min(activation.shape[1], source.shape[1])
            activation[:, :min_seq] = source[:, :min_seq]

        return activation

    def to_dict(self) -> dict:
        return {
            "type": "patch",
            "params": {"source_shape": list(self.source_activation.shape)},
        }

    @classmethod
    def _from_params(cls, params: dict) -> ActivationPatching:
        shape = params.get("source_shape", [1])
        return cls(source_activation=torch.zeros(shape))


class AdditivePerturbation(Intervention):
    """Add a fixed direction vector scaled by magnitude.

    Useful for directional probing: does adding a specific direction
    to the residual stream steer behavior in a predictable way?
    """

    def __init__(self, direction: torch.Tensor, magnitude: float = 1.0):
        self.direction = direction
        self.magnitude = magnitude

    def apply(self, activation: torch.Tensor, target: HookTarget) -> torch.Tensor:
        activation = activation.clone()
        direction = self.direction.to(activation.device, dtype=activation.dtype)
        perturbation = direction * self.magnitude

        if target.token_position is not None:
            activation[:, target.token_position] = (
                activation[:, target.token_position] + perturbation
            )
        else:
            activation = activation + perturbation

        return activation

    def to_dict(self) -> dict:
        return {
            "type": "additive",
            "params": {
                "magnitude": self.magnitude,
                "direction_shape": list(self.direction.shape),
            },
        }

    @classmethod
    def _from_params(cls, params: dict) -> AdditivePerturbation:
        shape = params.get("direction_shape", [1])
        return cls(
            direction=torch.zeros(shape),
            magnitude=params.get("magnitude", 1.0),
        )


def make_intervention_hook(
    target: HookTarget,
    intervention: Intervention,
) -> callable:
    """Create a hook function that applies an intervention.

    Returns a hook_fn or pre_hook_fn depending on the target type.
    """

    def hook_fn(module, input, output):
        tensor = output
        is_tuple = isinstance(tensor, tuple)
        if is_tuple:
            tensor = tensor[0]

        modified = intervention.apply(tensor, target)

        if is_tuple:
            return (modified,) + output[1:]
        return modified

    def pre_hook_fn(module, input):
        tensor = input
        is_tuple = isinstance(tensor, tuple)
        if is_tuple:
            tensor = tensor[0]

        modified = intervention.apply(tensor, target)

        if is_tuple:
            return (modified,) + input[1:]
        return (modified,)

    return pre_hook_fn if target.is_pre_hook else hook_fn
