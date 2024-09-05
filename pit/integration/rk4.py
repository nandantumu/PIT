import torch
from torch import nn
from ..parameters.definitions import AbstractParameterGroup
from ..parameters.point import PointParameterGroup

class RK4(nn.Module):
    """Module to do RK4 integration"""
    def __init__(self, dynamics,  parameters: AbstractParameterGroup=None, timestep=0.10, include_initial_state=False) -> None:
        super().__init__()
        self.dynamics = dynamics
        if parameters is None:
            self.model_params = PointParameterGroup(self.dynamics.parameter_list)
        else:
            self.model_params = parameters
        self.timestep = timestep
        self.include_initial_state = include_initial_state

    def forward(self, initial_state, control_inputs, time_deltas=None):
        """
        We integrate the specified dynamics
        
        Args:
            initial_state: Shape of (B, state_dims) or (state_dims)
            control_inputs: Shape of (B, L, input_dims) or (L, input_dims)
            time_deltas: Shape of (B, L) or (L)
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
            params = self.model_params.sample_parameters(B)
            if time_deltas is None:
                time_deltas = torch.ones((B, L), device=initial_state.device) * self.timestep
        else:
            L, _ = control_inputs.shape
            params = self.model_params.sample_parameters()
            if time_deltas is None:
                time_deltas = torch.ones((L), device=initial_state.device) * self.timestep

        integrated_states = list()

        if self.include_initial_state:
            integrated_states.append(initial_state)
        
        if batch_mode:
            k1 = self.dynamics(initial_state, control_inputs[:, 0], params)
            k2_state = initial_state + (time_deltas[:, 0] * k1 / 2)
            k2 = self.dynamics(k2_state, control_inputs[:,0], params)
            k3_state = initial_state + (time_deltas[:, 0] * k2 / 2)
            k3 = self.dynamics(k3_state, control_inputs[:,0], params)
            k4_state = initial_state + (time_deltas[:, 0] * k3)
            k4 = self.dynamics(k4_state, control_inputs[:,0], params)

            integrated_states.append(initial_state + (time_deltas[:, 0]*(k1 + 2*k2 + 2*k3 + k4)/6))

            for i in range(1, L):
                k1 = self.dynamics(integrated_states[i-1], control_inputs[:,i], params)
                k2_state = integrated_states[i-1] + (time_deltas[:, i] * k1 / 2)
                k2 = self.dynamics(integrated_states[i-1], control_inputs[:,i], params)
                k3_state = integrated_states[i-1] + (time_deltas[:, i] * k2 / 2)
                k3 = self.dynamics(integrated_states[i-1], control_inputs[:,i], params)
                k4_state = integrated_states[i-1] + (time_deltas[:, i] * k3)
                k4 = self.dynamics(integrated_states[i-1], control_inputs[:,i], params)
                integrated_states.append(integrated_states[i-1] + (time_deltas[:, i]*(k1 + 2*k2 + 2*k3 + k4)/6))
            
            integrated_states = torch.stack(integrated_states, dim=1)
            assert(list(integrated_states.shape) == [control_inputs.shape[0], control_inputs.shape[1], state_dims])
        
        else:
            k1 = self.dynamics(initial_state, control_inputs[0], params)
            k2_state = initial_state + (time_deltas[0] * k1 / 2)
            k2 = self.dynamics(k2_state, control_inputs[0], params)
            k3_state = initial_state + (time_deltas[0] * k2 / 2)
            k3 = self.dynamics(k3_state, control_inputs[0], params)
            k4_state = initial_state + (time_deltas[0] * k3)
            k4 = self.dynamics(k4_state, control_inputs[0], params)

            integrated_states.append(initial_state + (time_deltas[0]*(k1 + 2*k2 + 2*k3 + k4)/6))

            for i in range(1, L):
                k1 = self.dynamics(integrated_states[i-1], control_inputs[i], params)
                k2_state = integrated_states[i-1] + (time_deltas[i] * k1 / 2)
                k2 = self.dynamics(integrated_states[i-1], control_inputs[i], params)
                k3_state = integrated_states[i-1] + (time_deltas[i] * k2 / 2)
                k3 = self.dynamics(integrated_states[i-1], control_inputs[i], params)
                k4_state = integrated_states[i-1] + (time_deltas[i] * k3)
                k4 = self.dynamics(integrated_states[i-1], control_inputs[i], params)
                integrated_states.append(integrated_states[i-1] + (time_deltas[i]*(k1 + 2*k2 + 2*k3 + k4)/6))
            
            integrated_states = torch.stack(integrated_states, dim=0)
            assert(list(integrated_states.shape) == [control_inputs.shape[0], state_dims])
        
        return integrated_states