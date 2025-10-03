from __future__ import annotations

from .._compat import jnp

from . import Dynamics
from ._batching import ensure_batch
from ..parameters.definitions import ParameterSample


class Bicycle(Dynamics):
    """Kinematic bicycle model with the vehicle centre as the reference point."""

    def __init__(self, wheelbase: float) -> None:
        self.wb = jnp.array(wheelbase, dtype=jnp.float32)

    def forward(self, states, control_inputs):
        X, Y, THETA, V = 0, 1, 2, 3
        STEER, ACCEL = 0, 1

        states = jnp.asarray(states)
        control_inputs = jnp.asarray(control_inputs)
        states, unbatch_states = ensure_batch(states)
        control_inputs, _ = ensure_batch(control_inputs)

        diff_x = states[..., V] * jnp.cos(states[..., THETA])
        diff_y = states[..., V] * jnp.sin(states[..., THETA])
        diff_theta = states[..., V] * jnp.tan(control_inputs[..., STEER]) / self.wb
        diff_v = control_inputs[..., ACCEL]
        diff = jnp.stack([diff_x, diff_y, diff_theta, diff_v], axis=-1)
        return unbatch_states(diff)


def kinematic_bicycle(states, control_inputs, params: ParameterSample):
    X, Y, THETA, V, YAW = 0, 1, 2, 3, 4
    STEER, ACCEL = 0, 1

    states = jnp.asarray(states)
    control_inputs = jnp.asarray(control_inputs)

    beta = jnp.arctan(
        jnp.tan(states[..., THETA]) * (params["lr"] / (params["lf"] + params["lr"]))
    )

    diff_x = states[..., V] * jnp.cos(states[..., THETA] + beta)
    diff_y = states[..., V] * jnp.sin(states[..., THETA] + beta)
    diff_theta = control_inputs[..., STEER]
    diff_v = control_inputs[..., ACCEL]
    diff_yaw = jnp.cos(beta) * jnp.tan(states[..., THETA]) / (params["lf"] + params["lr"])
    return jnp.stack([diff_x, diff_y, diff_theta, diff_v, diff_yaw], axis=-1)
