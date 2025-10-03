"""Euler integration module."""

from __future__ import annotations

import torch
from torch import nn

from ..parameters.definitions import AbstractParameterGroup
from ..parameters.point import PointParameterGroup
from ._utils import _normalize_batch_inputs


class Euler(nn.Module):
    """Module to do Euler integration."""

    def __init__(
        self,
        dynamics,
        parameters: AbstractParameterGroup | None = None,
        timestep=0.10,
        include_initial_state=False,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        if parameters is None:
            self.model_params = PointParameterGroup(self.dynamics.parameter_list)
        else:
            self.model_params = parameters
        self.timestep = timestep
        self.include_initial_state = include_initial_state

    def forward(self, initial_state, control_inputs, time_deltas=None):
        """Integrate the dynamics using the Euler method."""

        (
            initial_state,
            control_inputs,
            time_deltas,
            params,
            was_batched,
        ) = _normalize_batch_inputs(
            initial_state,
            control_inputs,
            time_deltas,
            self.timestep,
            self.model_params,
        )

        current_state = initial_state
        integrated_states = [current_state]

        for i in range(control_inputs.shape[1]):
            dt = time_deltas[:, i].unsqueeze(1)
            control = control_inputs[:, i]
            diff = self.dynamics(current_state, control, params)
            current_state = current_state + diff * dt
            integrated_states.append(current_state)

        integrated_states = torch.stack(integrated_states, dim=1)

        if not self.include_initial_state:
            integrated_states = integrated_states[:, 1:]

        if not was_batched:
            integrated_states = integrated_states.squeeze(0)

        return integrated_states
