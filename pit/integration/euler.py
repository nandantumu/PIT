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
        input_dims = control_inputs.shape[-1]
        if batch_mode:
            B, L, _ = control_inputs.shape
        else:
            L, _ = control_inputs.shape

        integrated_states = list()

        if batch_mode:
            diff = self.dynamics(initial_state, control_inputs[:,0])
            #state = torch.zeros((B, state_dims))
            state = initial_state + diff * self.timestep
            integrated_states.append(state)

            for i in range(1, control_inputs.shape[1]):            
                #state = torch.zeros((B, state_dims))
                diff = self.dynamics(integrated_states[i-1], control_inputs[:,i])
                state = integrated_states[i-1] + diff * self.timestep
                integrated_states.append(state)
            integrated_states = torch.stack(integrated_states, dim=1)
            assert(list(integrated_states.shape) == [control_inputs.shape[0], control_inputs.shape[1], state_dims])
        
        else:
            diff = self.dynamics(initial_state, control_inputs[0])
            #state = torch.zeros((state_dims))
            state = initial_state + diff * self.timestep
            integrated_states.append(state)

            for i in range(1, control_inputs.shape[0]):
                diff = self.dynamics(integrated_states[i-1], control_inputs[i])
                #state = torch.zeros((state_dims))
                state = integrated_states[i-1] + diff * self.timestep
                integrated_states.append(state)
            
            integrated_states = torch.stack(integrated_states, dim=0)
            assert(list(integrated_states.shape) == [control_inputs.shape[0], state_dims])
        
        
        return integrated_states

