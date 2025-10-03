from . import Dynamics
from ._batching import ensure_batch
from ..parameters.definitions import ParameterSample

import torch
from torch import nn

class Bicycle(Dynamics, nn.Module):
    """
    This is a kinematic bicycle model, with the center of the vehicle as the control point. 
    Based on 
    https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html
    """
    def __init__(self, wheelbase) -> None:
        super().__init__()
        self.wb = torch.nn.Parameter(torch.tensor(wheelbase, dtype=torch.float32))

    def forward(self, states, control_inputs):
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
        diff[..., THETA] = (
            states[..., V] * torch.tan(control_inputs[..., STEER])
        ) / self.wb
        diff[..., V] = control_inputs[..., ACCEL]
        return unbatch_states(diff)


def kinematic_bicycle(states, control_inputs, params: ParameterSample):
    """Get the evaluated ODEs of the state at this point

    Args:
        states (): Shape of (B, 5) or (5)
        control_inputs (): Shape of (B, 2) or (2)
    """
    X, Y, THETA, V, YAW = 0, 1, 2, 3, 4
    STEER, ACCEL = 0, 1

    beta = torch.atan(torch.tan(states[..., THETA]) * (params['lr']/(params['lf']+params['lr'])))

    diff = torch.zeros_like(states)
    diff[..., X] = states[..., V]  * torch.cos(states[..., THETA] + beta)
    diff[..., Y] = states[..., V]  * torch.sin(states[..., THETA] + beta)
    diff[..., THETA] = control_inputs[..., STEER]
    diff[..., V] = control_inputs[..., ACCEL]
    diff[..., YAW] = torch.cos(beta) * torch.tan(states[..., THETA]) / (params['lf']+params['lr'])
    return diff