import torch
from torch import nn

class RK4(nn.Module):
    """Module to do RK4 integration"""
    def __init__(self, dynamics, timestep=0.10) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.timestep = timestep

    def forward(self, initial_state, control_inputs):
        """
        We integrate the specified dynamics
        
        Args:
            initial_state: Shape of (N,state_dims)
            control_inputs: Shape of (N,L,input_dims)
        Output:
            integrated_states: Shape of (N, L, state_dims)
        """
        
        pass