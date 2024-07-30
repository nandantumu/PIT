import torch
from torch import nn
from torch.nn import functional as F

class PointParameterGroup(nn.Module):
    def __init__(self, parameter_list: list, initial_value: dict=None):
        super().__init__()
        self.params = nn.ParameterDict()
        for param in parameter_list:
            self.params[param] = nn.Parameter(torch.tensor(initial_value[param] if initial_value else 0.0))
        
    def __getitem__(self, key):
        return self.params[key]
    
    def __setitem__(self, key, value):
        self.params[key] = nn.Parameter(torch.tensor(value))

    def forward(self):
        return self.params