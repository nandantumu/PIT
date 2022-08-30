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
            k1 = self.dynamics(initial_state, control_inputs[:,0])
            k2_state = initial_state + self.timestep * k1 / 2
            k2 = self.dynamics(k2_state, control_inputs[:,0])
            k3_state = initial_state + self.timestep * k2 / 2
            k3 = self.dynamics(k3_state, control_inputs[:,0])
            k4_state = initial_state + self.timestep * k3
            k4 = self.dynamics(k4_state, control_inputs[:,0])

            integrated_states.append(initial_state + self.timestep*(k1 + 2*k2 + 2*k3 + k4)/6)

            for i in range(1, control_inputs.shape[1]):
                k1 = self.dynamics(integrated_states[i-1], control_inputs[:,i])
                k2_state = integrated_states[i-1] + self.timestep * k1 / 2
                k2 = self.dynamics(integrated_states[i-1], control_inputs[:,i])
                k3_state = integrated_states[i-1] + self.timestep * k2 / 2
                k3 = self.dynamics(integrated_states[i-1], control_inputs[:,i])
                k4_state = integrated_states[i-1] + self.timestep * k3
                k4 = self.dynamics(integrated_states[i-1], control_inputs[:,i])
                integrated_states.append(initial_state + self.timestep*(k1 + 2*k2 + 2*k3 + k4)/6)
            
            integrated_states = torch.stack(integrated_states, dim=1)
            assert(list(integrated_states.shape) == [control_inputs.shape[0], control_inputs.shape[1], state_dims])
        
        else:
            k1 = self.dynamics(initial_state, control_inputs[0])
            k2_state = initial_state + self.timestep * k1 / 2
            k2 = self.dynamics(k2_state, control_inputs[0])
            k3_state = initial_state + self.timestep * k2 / 2
            k3 = self.dynamics(k3_state, control_inputs[0])
            k4_state = initial_state + self.timestep * k3
            k4 = self.dynamics(k4_state, control_inputs[0])

            integrated_states.append(initial_state + self.timestep*(k1 + 2*k2 + 2*k3 + k4)/6)

            for i in range(1, control_inputs.shape[0]):
                k1 = self.dynamics(integrated_states[i-1], control_inputs[i])
                k2_state = integrated_states[i-1] + self.timestep * k1 / 2
                k2 = self.dynamics(integrated_states[i-1], control_inputs[i])
                k3_state = integrated_states[i-1] + self.timestep * k2 / 2
                k3 = self.dynamics(integrated_states[i-1], control_inputs[i])
                k4_state = integrated_states[i-1] + self.timestep * k3
                k4 = self.dynamics(integrated_states[i-1], control_inputs[i])
                integrated_states.append(initial_state + self.timestep*(k1 + 2*k2 + 2*k3 + k4)/6)
            
            integrated_states = torch.stack(integrated_states, dim=0)
            assert(list(integrated_states.shape) == [control_inputs.shape[0], state_dims])
        
        return integrated_states