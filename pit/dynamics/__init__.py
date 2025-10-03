"""Base definitions for dynamics models."""

from __future__ import annotations

from ..parameters.definitions import ParameterSample


class Dynamics:
    """Base class for dynamics models."""

    parameter_list: list[str]

    def forward(self, states, inputs, params: ParameterSample):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, states, inputs, params: ParameterSample):
        return self.forward(states, inputs, params)
