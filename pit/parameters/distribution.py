import torch
from torch import nn
from torch.nn import functional as F

class IndependentNormalParameterGroup(nn.Module):
    def __init__(self, parameter_list: list, initial_value: dict=None):
        super().__init__()
        self.parameter_list = parameter_list
        self.params = nn.ParameterDict()
        for param in parameter_list:
            self.params[param+'_mean'] = nn.Parameter(torch.tensor(initial_value[param] if initial_value else 0.0))
            self.params[param+'_std'] = nn.Parameter(torch.tensor(initial_value[param+'_std'] if initial_value and param+'_std' in initial_value else 1.0))
            self.params[param] = torch.distributions.Normal(self.params[param+'_mean'], self.params[param+'_std'])
        
        
    def __getitem__(self, key):
        return self.params[key].rsample()
    
    def __setitem__(self, key, value):
        self.params[key] = nn.Parameter(torch.tensor(value))

    def forward(self):
        return {
            param: self.params[param+'_mean'] for param in self.parameter_list
        }