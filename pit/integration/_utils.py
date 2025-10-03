"""Shared utilities for integration modules."""

from __future__ import annotations

from typing import Any, Tuple

import torch


def _normalize_batch_inputs(
    initial_state: torch.Tensor,
    control_inputs: torch.Tensor,
    time_deltas: torch.Tensor | None,
    default_dt: float,
    parameter_group: Any | None = None,
    params_override: Any | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, bool]:
    """Normalize integration inputs to batched tensors.

    Args:
        initial_state: Tensor with shape ``(B, state_dims)`` or ``(state_dims,)``.
        control_inputs: Tensor with shape ``(B, steps, input_dims)`` or
            ``(steps, input_dims)``.
        time_deltas: Optional tensor with shape ``(B, steps)`` or ``(steps,)``.
        default_dt: Default time step to use when ``time_deltas`` is ``None``.
        parameter_group: Parameter group used to sample parameters, when
            available.
        params_override: Optional parameters provided by the caller.

    Returns:
        Tuple containing normalized tensors for ``initial_state``,
        ``control_inputs``, ``time_deltas``, the parameters used for dynamics,
        and a boolean flag indicating whether the original inputs were batched.
    """

    if control_inputs.ndim < 2:
        raise ValueError("Control inputs are not in the correct shape")

    was_batched = initial_state.ndim == 2

    if not was_batched:
        initial_state = initial_state.unsqueeze(0)
        control_inputs = control_inputs.unsqueeze(0)

    batch_size = initial_state.shape[0]
    steps = control_inputs.shape[1]

    if time_deltas is None:
        time_deltas = torch.full(
            (batch_size, steps),
            fill_value=default_dt,
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
    else:
        if time_deltas.ndim == 1:
            time_deltas = time_deltas.unsqueeze(0)
        if time_deltas.ndim != 2:
            raise ValueError("time_deltas must have shape (B, L) or (L,)")
        if time_deltas.shape[0] == 1 and batch_size != 1:
            time_deltas = time_deltas.expand(batch_size, -1)
        elif time_deltas.shape[0] != batch_size:
            raise ValueError("time_deltas batch dimension does not match inputs")
        if time_deltas.shape[1] != steps:
            raise ValueError("time_deltas step dimension does not match inputs")
        time_deltas = time_deltas.to(device=initial_state.device, dtype=initial_state.dtype)

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

