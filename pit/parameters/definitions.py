import torch
from torch import nn


class ParameterSample:
    def __init__(self, parameters: torch.Tensor, parameter_lookup: dict):
        self.parameters = parameters
        self.parameter_lookup = parameter_lookup

    def __getitem__(self, key):
        """This method should return the parameter value(s) for the key in text."""
        return self.parameters[self.parameter_lookup[key]]


class AbstractParameterGroup(nn.Module):
    def __init__(self, parameter_list: list, initial_value: dict=None):
        super().__init__()
        self.parameter_list = parameter_list
        self.parameter_lookup = {param: i for i, param in enumerate(parameter_list)}
        self.initialize_parameters()
        if initial_value:
            self.apply_initial_value(initial_value)
        
    @property
    def num_params(self):
        return len(self.parameter_list)

    def disable_gradients(self, parameter_name: str):
        """This function should disable gradients for the given parameter."""
        raise NotImplementedError
    
    def enable_gradients(self, parameter_name: str):
        """This function should enable gradients for the given parameter."""
        raise NotImplementedError

    def initialize_parameters(self):
        """This function should initialize the parameters of this object."""
        raise NotImplementedError

    def apply_initial_value(self, initial_value: dict):
        """This function should apply the initial parameter set to this object."""
        raise NotImplementedError

    def get_evaluation_sample(self, batch_size: int=1) -> ParameterSample:
        """
        This method should return a sample of the parameters that can be used for evaluation. 
        This may be stable over multiple calls.
        """
        raise NotImplementedError

    def sample_parameters(self, batch_size: int=1) -> ParameterSample:
        """
        This method should return a sample of the parameters that can be used for training.
        This could return a different sample each time it is called.
        """
        raise NotImplementedError
    
    def draw_parameters(self, batch_size: int=1) -> ParameterSample:
        """
        This method will call the appropriate method based on training state.
        """
        if self.training:
            return self.sample_parameters(batch_size)
        else:
            return self.get_evaluation_sample(batch_size)