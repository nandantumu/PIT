from __future__ import annotations

from .._compat import jnp

from . import Dynamics
from ._batching import ensure_batch


class Unicycle(Dynamics):
    """Kinematic unicycle model."""

    def __init__(self) -> None:
        self.parameter_list = ["null"]

    def forward(self, states, control_inputs, params):
        del params  # Unused for the unicycle model.
        X, Y, THETA, V = 0, 1, 2, 3
        STEER, ACCEL = 0, 1

        states = jnp.asarray(states)
        control_inputs = jnp.asarray(control_inputs)
        states, unbatch_states = ensure_batch(states)
        control_inputs, _ = ensure_batch(control_inputs)

        diff_x = states[..., V] * jnp.cos(states[..., THETA])
        diff_y = states[..., V] * jnp.sin(states[..., THETA])
        diff_theta = control_inputs[..., STEER]
        diff_v = control_inputs[..., ACCEL]
        diff = jnp.stack([diff_x, diff_y, diff_theta, diff_v], axis=-1)
        return unbatch_states(diff)
