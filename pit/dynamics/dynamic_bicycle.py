from __future__ import annotations

from .._compat import jnp

from . import Dynamics
from ._batching import ensure_batch
from ..parameters.definitions import ParameterSample

X, Y, YAW, VX, VY, YAW_RATE, STEERING_ANGLE = 0, 1, 2, 3, 4, 5, 6
DRIVE_FORCE, STEER_SPEED = 0, 1
FRX, FFY, FRY = 0, 1, 2


class DynamicBicycle(Dynamics):
    """Dynamic bicycle model based on the AMZ Driverless formulation."""

    def __init__(self, lf, lr, Iz, m, Df, Cf, Bf, Dr, Cr, Br, Cm, Cr0, Cr2, **kwargs) -> None:
        del kwargs
        self.parameter_list = [
            "lf",
            "lr",
            "Iz",
            "m",
            "Df",
            "Cf",
            "Bf",
            "Dr",
            "Cr",
            "Br",
            "Cm",
            "Cr0",
            "Cr2",
        ]
        self.initial_values = {
            "lf": lf,
            "lr": lr,
            "Iz": Iz,
            "m": m,
            "Df": Df,
            "Cf": Cf,
            "Bf": Bf,
            "Dr": Dr,
            "Cr": Cr,
            "Br": Br,
            "Cm": Cm,
            "Cr0": Cr0,
            "Cr2": Cr2,
        }

    def calculate_tire_forces(self, states, control_inputs, params: ParameterSample):
        states = jnp.asarray(states)
        control_inputs = jnp.asarray(control_inputs)

        states, unbatch_states = ensure_batch(states)
        control_inputs, _ = ensure_batch(control_inputs)

        alpha_f = states[..., STEERING_ANGLE] - jnp.arctan(
            (states[..., YAW_RATE] * params["lf"] + states[..., VY]) / states[..., VX]
        )
        alpha_r = jnp.arctan(
            (states[..., YAW_RATE] * params["lr"] - states[..., VY]) / states[..., VX]
        )

        frx = (
            params["Cm"] * control_inputs[..., DRIVE_FORCE]
            - params["Cr0"]
            - params["Cr2"] * states[..., VX] ** 2.0
        )
        ffy = params["Df"] * jnp.sin(params["Cf"] * jnp.arctan(params["Bf"] * alpha_f))
        fry = params["Dr"] * jnp.sin(params["Cr"] * jnp.arctan(params["Br"] * alpha_r))
        tire_forces = jnp.stack([frx, ffy, fry], axis=-1)
        return unbatch_states(tire_forces)

    def forward(self, states, control_inputs, params: ParameterSample):
        states = jnp.asarray(states)
        control_inputs = jnp.asarray(control_inputs)

        states, unbatch_states = ensure_batch(states)
        control_inputs, _ = ensure_batch(control_inputs)

        tire_forces = self.calculate_tire_forces(states, control_inputs, params)

        diff_x = (
            states[..., VX] * jnp.cos(states[..., YAW])
            - states[..., VY] * jnp.sin(states[..., YAW])
        )
        diff_y = (
            states[..., VX] * jnp.sin(states[..., YAW])
            - states[..., VY] * jnp.cos(states[..., YAW])
        )
        diff_yaw = states[..., YAW_RATE]
        diff_vx = (
            tire_forces[..., FRX]
            - tire_forces[..., FFY] * jnp.sin(states[..., STEERING_ANGLE])
            + states[..., VY] * states[..., YAW_RATE] * params["m"]
        ) / params["m"]
        diff_vy = (
            tire_forces[..., FRY]
            + tire_forces[..., FFY] * jnp.cos(states[..., STEERING_ANGLE])
            - states[..., VX] * states[..., YAW_RATE] * params["m"]
        ) / params["m"]
        diff_yaw_rate = (
            tire_forces[..., FFY] * params["lf"] * jnp.cos(states[..., STEERING_ANGLE])
            - tire_forces[..., FRY] * params["lr"]
        ) / params["Iz"]
        diff_steer = control_inputs[..., STEER_SPEED]

        diff = jnp.stack(
            [
                diff_x,
                diff_y,
                diff_yaw,
                diff_vx,
                diff_vy,
                diff_yaw_rate,
                diff_steer,
            ],
            axis=-1,
        )
        return unbatch_states(diff)
