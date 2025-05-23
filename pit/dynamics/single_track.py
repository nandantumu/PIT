from . import Dynamics
from ..parameters.definitions import ParameterSample

import torch
from torch import nn


class SingleTrack(Dynamics, nn.Module):
    """
    This is the Single Track model, from the CommonRoad paper.
    Link: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf
    """

    def __init__(self, m, Iz, lf, lr, hcg, Csf, Csr, mu, **kwargs) -> None:
        super().__init__()
        self.parameter_list = ["m", "Iz", "lf", "lr", "hcg", "Csf", "Csr", "mu"]
        self.initial_values = {
            "m": m,
            "Iz": Iz,
            "lf": lf,
            "lr": lr,
            "hcg": hcg,
            "Csf": Csf,
            "Csr": Csr,
            "mu": mu,
        }
        self.g = 9.81
        self.numeric_stability_constant = 1e-10

    def forward(self, states, control_inputs, params: ParameterSample):
        """Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 6) or (6)
                [X, Y, V, YAW, YAW_RATE, SLIP_ANGLE]
            control_inputs (): Shape of (B, 2) or (2)
                [STEER_ANGLE, ACCEL]
        """
        batch_mode = True if len(states.shape) == 2 else False
        X, Y, V, YAW, YAW_RATE, SLIP_ANGLE = 0, 1, 2, 3, 4, 5
        CONTROL_STEER_ANGLE, ACCEL = 0, 1
        diff = torch.zeros_like(states)
        diff[..., X] = states[..., V] * torch.cos(
            states[..., YAW] + states[..., SLIP_ANGLE]
        )
        diff[..., Y] = states[..., V] * torch.sin(
            states[..., YAW] + states[..., SLIP_ANGLE]
        )
        diff[..., YAW] = states[..., YAW_RATE]
        diff[..., V] = control_inputs[..., ACCEL]
        glr = self.g * params["lr"] - control_inputs[..., ACCEL] * params["hcg"]
        glf = self.g * params["lf"] + control_inputs[..., ACCEL] * params["hcg"]
        diff[..., YAW_RATE] = (
            (params["mu"] * params["m"])
            / (params["Iz"] * (params["lf"] + params["lr"]))
        ) * (
            params["lf"]
            * params["Csf"]
            * glr
            * control_inputs[..., CONTROL_STEER_ANGLE]
            + (params["lr"] * params["Csr"] * glf - params["lf"] * params["Csf"] * glr)
            * states[..., SLIP_ANGLE]
            - (
                params["lf"] * params["lf"] * params["Csf"] * glr
                + params["lr"] * params["lr"] * params["Csr"] * glf
            )
            * (
                states[..., YAW_RATE]
                / (self.numeric_stability_constant + states[..., V])
            )
        )

        diff[..., SLIP_ANGLE] = (
            params["mu"] / (states[..., V] * (params["lr"] + params["lf"]))
        ) * (
            params["Csf"] * glr * control_inputs[..., CONTROL_STEER_ANGLE]
            - (params["Csr"] * glf + params["Csf"] * glr) * states[..., SLIP_ANGLE]
            + (params["Csr"] * glf * params["lr"] - params["Csf"] * glr * params["lf"])
            * (
                states[..., YAW_RATE]
                / (self.numeric_stability_constant + states[..., V])
            )
        ) - states[..., YAW_RATE]

        return diff
