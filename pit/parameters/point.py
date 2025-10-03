"""Deterministic parameter group backed by JAX arrays."""

from __future__ import annotations

from typing import Dict

from .._compat import jnp

from .definitions import AbstractParameterGroup, ParameterSample


class PointParameterGroup(AbstractParameterGroup):
    """Parameter group whose values are simple point estimates."""

    def __init__(self, parameter_list: list, initial_value: dict | None = None):
        super().__init__(parameter_list, initial_value)

    def initialize_parameters(self) -> None:
        self.params: Dict[str, jnp.ndarray] = {
            name: jnp.array(0.0, dtype=jnp.float32) for name in self.parameter_list
        }

    def apply_initial_value(self, initial_value: dict) -> None:
        for name, value in initial_value.items():
            if name in self.params:
                self.params[name] = jnp.asarray(value, dtype=jnp.float32)

    def _stack_params(self) -> jnp.ndarray:
        if not self.parameter_list:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.stack([self.params[name] for name in self.parameter_list])

    def get_evaluation_sample(self, batch_size: int = 1) -> ParameterSample:
        values = self._stack_params()
        if values.ndim == 0:
            values = values[None]
        values = jnp.broadcast_to(values[:, None], (values.shape[0], batch_size))
        return ParameterSample(values, self.parameter_lookup)

    def sample_parameters(self, batch_size: int = 1) -> ParameterSample:
        return self.get_evaluation_sample(batch_size)
