from __future__ import annotations

from typing import Union

from .._compat import jnp
from ..parameters.definitions import AbstractParameterGroup
from ..parameters.point import PointParameterGroup
from ._utils import _normalize_batch_inputs


class RK4:
    """Runge-Kutta 4th order integrator implemented with JAX arrays."""

    def __init__(
        self,
        dynamics,
        parameters: Union[AbstractParameterGroup, str, None] = None,
        timestep: float = 0.10,
        include_initial_state: bool = False,
    ) -> None:
        self.dynamics = dynamics
        if parameters == "BYPASS":
            self.model_params = "BYPASS"
        elif parameters is None:
            self.model_params = PointParameterGroup(self.dynamics.parameter_list)
        else:
            self.model_params = parameters
        self.timestep = timestep
        self.include_initial_state = include_initial_state

    def forward(self, initial_state, control_inputs, time_deltas=None, params=None):
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
            params,
        )

        current_state = initial_state
        integrated_states = [current_state]

        for i in range(control_inputs.shape[1]):
            dt = jnp.expand_dims(time_deltas[:, i], axis=1)
            control = control_inputs[:, i]

            k1 = self.dynamics(current_state, control, params)
            k2_state = current_state + dt * k1 / 2.0
            k2 = self.dynamics(k2_state, control, params)
            k3_state = current_state + dt * k2 / 2.0
            k3 = self.dynamics(k3_state, control, params)
            k4_state = current_state + dt * k3
            k4 = self.dynamics(k4_state, control, params)

            current_state = current_state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            integrated_states.append(current_state)

        integrated_states = jnp.stack(integrated_states, axis=1)

        if not self.include_initial_state:
            integrated_states = integrated_states[:, 1:]

        if not was_batched:
            integrated_states = jnp.squeeze(integrated_states, axis=0)

        return integrated_states

    __call__ = forward
