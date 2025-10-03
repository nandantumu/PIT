"""Shared utilities for integration modules."""

from __future__ import annotations

from typing import Any, Tuple

from .._compat import jnp


def _normalize_batch_inputs(
    initial_state,
    control_inputs,
    time_deltas,
    default_dt: float,
    parameter_group: Any | None = None,
    params_override: Any | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any, bool]:
    """Normalize integration inputs to batched arrays."""

    initial_state = jnp.asarray(initial_state)
    control_inputs = jnp.asarray(control_inputs)

    if control_inputs.ndim < 2:
        raise ValueError("Control inputs are not in the correct shape")

    was_batched = initial_state.ndim == 2

    if not was_batched:
        initial_state = jnp.expand_dims(initial_state, axis=0)
        control_inputs = jnp.expand_dims(control_inputs, axis=0)

    batch_size = initial_state.shape[0]
    steps = control_inputs.shape[1]

    if time_deltas is None:
        time_deltas = jnp.full((batch_size, steps), default_dt, dtype=initial_state.dtype)
    else:
        time_deltas = jnp.asarray(time_deltas, dtype=initial_state.dtype)
        if time_deltas.ndim == 1:
            time_deltas = jnp.expand_dims(time_deltas, axis=0)
        if time_deltas.ndim != 2:
            raise ValueError("time_deltas must have shape (B, L) or (L,)")
        if time_deltas.shape[0] == 1 and batch_size != 1:
            time_deltas = jnp.broadcast_to(time_deltas, (batch_size, time_deltas.shape[1]))
        elif time_deltas.shape[0] != batch_size:
            raise ValueError("time_deltas batch dimension does not match inputs")
        if time_deltas.shape[1] != steps:
            raise ValueError("time_deltas step dimension does not match inputs")

    params = None
    if params_override == "BYPASS":
        params = parameter_group
    elif params_override is not None:
        params = params_override
    elif parameter_group == "BYPASS":
        params = parameter_group
    elif parameter_group is not None:
        if was_batched:
            params = parameter_group.draw_parameters(batch_size)
        else:
            params = parameter_group.draw_parameters()

    return initial_state, control_inputs, time_deltas, params, was_batched
