import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import register_parametrization
from .definitions import AbstractParameterGroup, ParameterSample
from torch.distributions import MultivariateNormal, Normal


class ScaleTril(nn.Module):
    """Module to ensure that a matrix is symmetric positive definite"""
    def forward(self, matrix, n):
        # Return a positive triangular matrix, with positive diagonal
        matrix = matrix @ matrix.T
        matrix = matrix + torch.diag(F.softplus(n))
        return torch.linalg.cholesky(matrix)

class Positive(nn.Module):
    """Module to ensure that a parameter is non-negative"""
    def forward(self, x):
        return F.relu(x) + 1e-6

class CovariantNormalParameterGroup(AbstractParameterGroup):
    """This class represents a group of parameters that are drawn from a multivariate normal distribution."""
    def __init__(self, parameter_list: list, initial_value: dict=None):
        super().__init__(parameter_list, initial_value)

    def initialize_parameters(self):
        self.loc = nn.Parameter(torch.zeros(len(self.parameter_list)))
        self.raw_covariance = nn.Parameter(torch.eye(len(self.parameter_list))+0.5)
        self.n = nn.Parameter(torch.ones(len(self.parameter_list)))
        self.scale_tril = ScaleTril()
        # self.scale_tril = nn.Parameter(torch.cholesky(covariance))
        # self.distribution = MultivariateNormal(loc=self.loc, covariance_matrix=self.covariance)

    @property
    def covariance(self):
        st = self.scale_tril(self.raw_covariance, self.n)
        return st @ st.T
    
    def apply_initial_value(self, initial_value: dict):
        self.loc.data = torch.tensor([initial_value[item] if item in initial_value else 0.0 for item in self.parameter_list])
        if 'covariance' in initial_value:
            self.raw_covariance.data = torch.tensor(initial_value['covariance'])
        else:
            self.raw_covariance.data = torch.eye(len(self.parameter_list))+0.5
        # self.distribution = torch.distributions.MultivariateNormal(self.loc, self.covariance)

    def get_evaluation_sample(self, batch_size: int=1):
        return ParameterSample(
            torch.tile(self.loc.unsqueeze(1), (1, batch_size)),
            self.parameter_lookup
        )
    
    def sample_parameters(self, batch_size: int=1):
        return ParameterSample(
            MultivariateNormal(loc=self.loc, scale_tril=self.scale_tril(self.raw_covariance, self.n)).rsample((batch_size, )).T,
            self.parameter_lookup
        )
    
    # def to(self, *args, **kwargs):
    #     super().to(*args, **kwargs)
    #     # self.distribution = torch.distributions.MultivariateNormal(self.loc, self.covariance)
    #     self.distribution._unbroadcasted_scale_tril = self.distribution._unbroadcasted_scale_tril.to(*args, **kwargs)
    #     self.distribution.loc = self.distribution.loc.to(*args, **kwargs)
    #     if self.distribution.scale_tril is not None:
    #         self.distribution.scale_tril = self.distribution.scale_tril.to(*args, **kwargs)
    #     if self.distribution.covariance_matrix is not None:
    #         self.distribution.covariance_matrix = self.distribution.covariance_matrix.to(*args, **kwargs)
    #     if self.distribution.precision_matrix is not None:
    #         self.distribution.precision_matrix = self.distribution.precision_matrix.to(*args, **kwargs)

# class CovariantNormalParameterGroup(nn.Module):
#     def __init__(self, parameter_list: list, initial_value: dict=None):
#         super().__init__()
#         self.parameter_list = parameter_list
#         self.parameter_lookup = {param: i for i, param in enumerate(parameter_list)}
#         self.num_params = len(parameter_list)

#         self.loc = nn.Parameter(torch.tensor([initial_value[item] if initial_value and item in initial_value else 0.0 for item in parameter_list]))
#         self.covariance = nn.Parameter(torch.eye(self.num_params))
#         self.distribution = torch.distributions.MultivariateNormal(self.loc, self.covariance)
#         # self.refresh_sample()
               
#     def __getitem__(self, key):
#         return self.distribution.rsample()[self.parameter_lookup[key]].to(self.loc.device)
    
#     def refresh_sample(self):
#         self.sample = self.distribution.rsample().to(self.loc.device)

#     # def __setitem__(self, key, value):
#     #     self.params[key] = nn.Parameter(torch.tensor(value))

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         # self.distribution = torch.distributions.MultivariateNormal(self.loc, self.covariance)
#         self.distribution._unbroadcasted_scale_tril = self.distribution._unbroadcasted_scale_tril.to(*args, **kwargs)
#         self.distribution.loc = self.distribution.loc.to(*args, **kwargs)
#         if self.distribution.scale_tril is not None:
#             self.distribution.scale_tril = self.distribution.scale_tril.to(*args, **kwargs)
#         if self.distribution.covariance_matrix is not None:
#             self.distribution.covariance_matrix = self.distribution.covariance_matrix.to(*args, **kwargs)
#         if self.distribution.precision_matrix is not None:
#             self.distribution.precision_matrix = self.distribution.precision_matrix.to(*args, **kwargs)

#     # def forward(self):
#     #     return {
#     #         param: self.loc[self.parameter_lookup[param]] for param in self.parameter_list
#     #     }
    
class NormalParameterGroup(AbstractParameterGroup):
    """This class represents a group of parameters that are drawn from a normal distribution."""
    def __init__(self, parameter_list: list, initial_value: dict=None):
        super().__init__(parameter_list, initial_value)

    def initialize_parameters(self):
        self.loc = nn.Parameter(torch.zeros(len(self.parameter_list)))
        self.positive = Positive()
        self.raw_scale = nn.Parameter(torch.ones(len(self.parameter_list)))
    
    @property
    def scale(self):
        return self.positive(self.raw_scale)

    def apply_initial_value(self, initial_value: dict):
        self.loc.data = torch.tensor([initial_value[item] if item in initial_value else 0.0 for item in self.parameter_list])
        self.raw_scale.data = torch.tensor([initial_value[item+"_scale"] if item+"_scale" in initial_value else 1.0 for item in self.parameter_list])

    def get_evaluation_sample(self, batch_size: int=1):
        return ParameterSample(
            torch.tile(self.loc.unsqueeze(1), (1, batch_size)),
            self.parameter_lookup
        )
    
    def sample_parameters(self, batch_size: int=1):
        return ParameterSample(
            Normal(self.loc, self.scale).rsample((batch_size,)).T,
            self.parameter_lookup
        )
    
    # def to(self, *args, **kwargs):
    #     super().to(*args, **kwargs)
    #     # self.distribution = torch.distributions.Normal(self.loc, self.scale)
    #     self.distribution.loc = self.distribution.loc.to(*args, **kwargs)
    #     self.distribution.scale = self.distribution.scale.to(*args, **kwargs)
