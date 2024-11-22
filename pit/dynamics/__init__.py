import torch
from torch import nn
from ..parameters.definitions import ParameterSample

class Dynamics(nn.Module):
    """Base Class for dynamics"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, states, inputs, params: ParameterSample):
        """
        Dynamics evolutions

        Args:
            states: Dimension of (N, state_dims)
            inputs: Dimension of (N, control_inputs)
        """
        raise NotImplementedError
