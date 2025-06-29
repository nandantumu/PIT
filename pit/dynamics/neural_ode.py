from . import Dynamics
from ..parameters.definitions import ParameterSample

import torch
from torch import nn


class NeuralODE(Dynamics, nn.Module):
    """
    This is the Single Track model, from the CommonRoad paper.
    Link: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf
    """

    def __init__(
        self, state_dims, control_dims, layers=5, hidden_dims=10, **kwargs
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dims + control_dims, hidden_dims),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU(),
                )
                for _ in range(layers)
            ],
            nn.Linear(hidden_dims, state_dims),
        )

    def forward(self, states, control_inputs, *args, **kwargs):
        """Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, S_d) or (S_d)
                [X, Y, V, YAW, YAW_RATE, SLIP_ANGLE]
            control_inputs (): Shape of (B, C_d) or (C_d)
                [STEER_ANGLE, ACCEL]
        """
        # Get the parameters for the neural network from the parameter sample
        input = torch.cat([states, control_inputs], dim=-1)
        # hidden_dims = self.params["layer_0_w"] @ input + self.params["layer_0_b"]
        # for i in range(1, len(self.params) // 2):
        #     hidden_dims = torch.relu(
        #         self.params[f"layer_{i}_w"] @ hidden_dims + self.params[f"layer_{i}_b"]
        #     )
        # output = self.params["layer_out_w"] @ hidden_dims + self.params["layer_out_b"]
        output = self.network(input)
        return output
