"""Minimal utilities for working with parameter groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .._compat import jnp


@dataclass(frozen=True)
class ParameterSample:
    """Dictionary-like container for parameter arrays."""

    parameters: jnp.ndarray
    parameter_lookup: Dict[str, int]

    def __getitem__(self, key: str) -> jnp.ndarray:
        return self.parameters[self.parameter_lookup[key]]


class AbstractParameterGroup:
    """Tiny base class for parameter groups backed by JAX arrays."""

    def __init__(
        self,
        parameter_list: Iterable[str],
        initial_value: Dict[str, Any] | None = None,
    ) -> None:
        self.parameter_list = list(parameter_list)
        self.parameter_lookup = {name: i for i, name in enumerate(self.parameter_list)}
        self.initialize_parameters()
        if initial_value:
            self.apply_initial_value(initial_value)

    @property
    def num_params(self) -> int:
        return len(self.parameter_list)

    # The following methods define the required API for concrete subclasses.
    def initialize_parameters(self) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def apply_initial_value(self, initial_value: Dict[str, Any]) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def get_evaluation_sample(self, batch_size: int = 1) -> ParameterSample:  # pragma: no cover - interface definition
        raise NotImplementedError

    def sample_parameters(self, batch_size: int = 1) -> ParameterSample:  # pragma: no cover - interface definition
        raise NotImplementedError

    # Backwards compatibility helpers -------------------------------------------------
    def train(self, mode: bool = True) -> "AbstractParameterGroup":  # pragma: no cover - API compat
        del mode
        return self

    def eval(self) -> "AbstractParameterGroup":  # pragma: no cover - API compat
        return self

    def disable_gradients(self, parameter_name: str) -> None:  # pragma: no cover - API compat
        del parameter_name

    def enable_gradients(self, parameter_name: str) -> None:  # pragma: no cover - API compat
        del parameter_name

    def draw_parameters(self, batch_size: int = 1) -> ParameterSample:
        return self.sample_parameters(batch_size)
