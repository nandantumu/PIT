from . import Dynamics

import torch
from torch import nn


class BicycleCoG(Dynamics, nn.Module):
    """
    This is a kinematic model with reference point in a center of gravity
    From common roads: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/tree/master/
    Model reference point: CoG
    State Variable [x, y, steering angle, v, yaw]
    Control Inputs [steering velocity, acceleration]
    """

    def __init__(self, lr, lf) -> None:
        super().__init__()
        self.lr = torch.nn.Parameter(torch.tensor(lr, dtype=torch.float32))
        self.lf = torch.nn.Parameter(torch.tensor(lf, dtype=torch.float32))

    def forward(self, states, control_inputs):
        """ Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 5) or (5)
            control_inputs (): Shape of (B, 2) or (2)
        """
        batch_mode = True if len(states.shape) == 2 else False
        X, Y, STEERING_ANGLE, V, THETA = 0, 1, 2, 3, 4
        STEER_V, ACCEL = 0, 1
        diff = torch.zeros_like(states)
        if batch_mode:
            pass
        else:

            l_wb = self.lf + self.lr

            # slip angle (beta) from vehicle kinematics
            beta = torch.atan(torch.tan(states[STEERING_ANGLE]) * self.lr / l_wb)

            diff[X] = states[V] * torch.cos(beta + states[THETA])
            diff[Y] = states[V] * torch.sin(beta + states[THETA])
            diff[STEERING_ANGLE] = control_inputs[STEER_V]
            diff[V] = control_inputs[ACCEL]
            diff[THETA] = states[V] * torch.cos(beta) * torch.tan(states[STEERING_ANGLE]) / l_wb

        return diff
