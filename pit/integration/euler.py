import torch
from torch import nn

class Euler(nn.Module):
    """Module to do Euler integration"""
    def __init__(self, dynamics, timestep=0.10) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.timestep = timestep

    def forward(self, initial_state, control_inputs):
        """
        We integrate the specified dynamics
        
        Args:
            initial_state: Shape of (B, state_dims) or (state_dims)
            control_inputs: Shape of (B, L, input_dims) or (L, input_dims)
        Output:
            integrated_states: Shape of (B, L, state_dims) or (L, state_dims)
        """
        batch_mode = True if len(initial_state.shape)==2 else False
        try:
            assert(len(control_inputs.shape) >= 2)
        except AssertionError:
            raise ValueError("Control inputs are not in the correct shape") 
        
        state_dims = initial_state.shape[-1]
        output_shape = list(control_inputs.shape)
        output_shape[-1] = state_dims
        integrated_states = torch.zeros(output_shape)

        if batch_mode:
            print("BATCHMODE")
            diff = self.dynamics(initial_state, control_inputs[:,0])
            integrated_states[:,0] = initial_state + diff * self.timestep

            for i in range(1, control_inputs.shape[1]):
                diff = self.dynamics(integrated_states[:,i-1], control_inputs[:,i])
                integrated_states[:,i] = integrated_states[:,i-1] + diff * self.timestep
        
        else:
            print("SINGLEMODE")
            diff = self.dynamics(initial_state, control_inputs[0])
            integrated_states[0] = initial_state + diff * self.timestep

            for i in range(1, control_inputs.shape[0]):
                diff = self.dynamics(integrated_states[i-1], control_inputs[i])
                integrated_states[i] = integrated_states[i-1] + diff * self.timestep
        
        return integrated_states

