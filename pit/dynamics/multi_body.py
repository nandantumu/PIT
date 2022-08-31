from . import Dynamics

import torch
from torch import nn

class Bicycle(Dynamics, nn.Module):
    """
    This is a kinematic bicycle model. Details from 
    https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html
    """
    def __init__(self, wheelbase) -> None:
        super().__init__()
        self.wb = torch.nn.Parameter(torch.tensor(wheelbase, dtype=torch.float32))
        raise NotImplementedError

    def forward(self, states, control_inputs):
        """ Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 4) or (4)
            control_inputs (): Shape of (B, 2) or (2)
        """
        batch_mode = True if len(states.shape)==2 else False
        X, Y, THETA, V = 0, 1, 2, 3
        STEER, ACCEL = 0, 1
        diff = torch.zeros_like(states)
        if batch_mode:
            diff[:, X] = states[:, V] * torch.cos(states[:, THETA])
            diff[:, Y] = states[:, V] * torch.sin(states[:, THETA])
            diff[:, THETA] = (states[:, V] * torch.tan(control_inputs[:, STEER]))/self.wb
            diff[:, V] = control_inputs[:, ACCEL]
        else:
            diff[X] = states[V] * torch.cos(states[THETA])
            diff[Y] = states[V] * torch.sin(states[THETA])
            diff[THETA] = (states[V] * torch.tan(control_inputs[STEER]))/self.wb
            diff[V] = control_inputs[ACCEL]
        return diff