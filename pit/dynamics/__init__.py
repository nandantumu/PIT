import torch
from torch import nn

class Dynamics(nn.Module):
    """Base Class for dynamics"""
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, states, inputs):
        """
        Dynamics evolutions

        Args:
            states: Dimension of (N, state_dims)
            inputs: Dimension of (N, control_inputs)
        """
        raise NotImplementedError
