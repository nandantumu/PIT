import torch
from torch import nn
from torch.nn import functional as F
from .definitions import AbstractParameterGroup, ParameterSample


class PointParameterGroup(AbstractParameterGroup):
    """This class represents a group of parameters that are point estimates."""
    def __init__(self, parameter_list: list, initial_value: dict=None):
        super().__init__(parameter_list, initial_value)

    def initialize_parameters(self):
        self.params = nn.ParameterDict()
        for param in self.parameter_list:
            self.params[param] = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def disable_gradients(self, parameter_name: str):
        self.params[parameter_name].requires_grad = False

    def enable_gradients(self, parameter_name: str):
        self.params[parameter_name].requires_grad = True
    
    def apply_initial_value(self, initial_value: dict):
        for param in self.parameter_list:
            self.params[param].data = torch.tensor(initial_value[param] if param in initial_value else 0.0, dtype=torch.float64)
    
    def get_evaluation_sample(self, batch_size: int=1):
        return ParameterSample(
            torch.tile(torch.stack([self.params[param].data for param in self.parameter_list]).reshape(-1, 1), (1, batch_size)),
            self.parameter_lookup
        )
    
    def sample_parameters(self, batch_size: int=1):
        return ParameterSample(
            torch.tile(torch.stack([self.params[param] for param in self.parameter_list]).reshape(-1, 1), (1, batch_size)),
            self.parameter_lookup
        )


# class PointParameterGroup(nn.Module):
#     def __init__(self, parameter_list: list, initial_value: dict=None):
#         super().__init__()
#         self.params = nn.ParameterDict()
#         for param in parameter_list:
#             self.params[param] = nn.Parameter(torch.tensor(initial_value[param] if initial_value else 0.0))
        
#     def __getitem__(self, key):
#         return self.params[key]
    
#     def __setitem__(self, key, value):
#         self.params[key] = nn.Parameter(torch.tensor(value))

#     def forward(self):
#         return self.params
    
# class PointParameter(nn.Module):
#     def __init__(self, initial_value: float=0.0):
#         super().__init__()
#         self.param = nn.Parameter(torch.tensor(initial_value))
        
#     def __getitem__(self, key):
#         return self.param
    
#     def __setitem__(self, key, value):
#         self.param = nn.Parameter(torch.tensor(value))

#     def forward(self):
#         return self.param