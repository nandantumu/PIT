from . import Dynamics
from ._batching import ensure_batch

import torch
from torch import nn

class Unicycle(Dynamics, nn.Module):
    """
    This is a kinematic Unicycle model.
    """
    def __init__(self) -> None:
        super().__init__()
        self.parameter_list = ['null']

    def forward(self, states, control_inputs, params):
        """ Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 4) or (4)
            control_inputs (): Shape of (B, 2) or (2)
        """
        X, Y, THETA, V = 0, 1, 2, 3
        STEER, ACCEL = 0, 1
        states, unbatch_states = ensure_batch(states)
        control_inputs, _ = ensure_batch(control_inputs)

        diff = torch.zeros_like(states)
        diff[..., X] = states[..., V] * torch.cos(states[..., THETA])
        diff[..., Y] = states[..., V] * torch.sin(states[..., THETA])
        diff[..., THETA] = control_inputs[..., STEER]
        diff[..., V] = control_inputs[..., ACCEL]
        return unbatch_states(diff)

