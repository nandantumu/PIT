import torch
from torch import nn
from torch.nn import functional as F
from .definitions import AbstractParameterGroup, ParameterSample


class BoundedParameterGroup(AbstractParameterGroup):
    """This class represents a group of parameters that are bounded."""

    def __init__(
        self, parameter_list: list, initial_value: dict = None, bounds: dict = None
    ):
        self.bounds = bounds
        super().__init__(parameter_list, initial_value)

    def initialize_parameters(self):
        self.params = nn.ParameterDict()
        for param in self.parameter_list:
            self.params[param] = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def apply_initial_value(self, initial_value: dict):
        for param in self.parameter_list:
            if param in initial_value:
                lower, upper = self.bounds[param]
                # We assume that the initial value is within the bounds
                assert (
                    initial_value[param] >= lower and initial_value[param] <= upper
                ), (
                    f"Initial value {initial_value[param]} for parameter {param} is out of bounds [{lower}, {upper}]"
                )
                # What is the value of the parameter when normalized to where -1 is lower and 1 is upper?
                normalized_value = torch.atanh(
                    ((torch.tensor(initial_value[param]) - lower) / (upper - lower)) - 1
                )

                self.params[param].data = torch.tensor(
                    normalized_value,
                    dtype=torch.float64,
                )
            else:
                # If the parameter is not in the initial value, we set it to 0
                self.params[param].data = torch.tensor(
                    0.0,
                    dtype=torch.float64,
                )

    def get_evaluation_sample(self, batch_size: int = 1):
        return ParameterSample(
            torch.tile(
                torch.stack(
                    [
                        (self.bounds[param][1] - self.bounds[param][0])
                        * (F.tanh(self.params[param]) + 1)
                        + self.bounds[param][0]
                        for param in self.parameter_list
                    ]
                ).reshape(-1, 1),
                (1, batch_size),
            ),
            self.parameter_lookup,
        )

    def sample_parameters(self, batch_size: int = 1):
        return ParameterSample(
            torch.tile(
                torch.stack(
                    [
                        (self.bounds[param][1] - self.bounds[param][0])
                        * (F.tanh(self.params[param]) + 1)
                        + self.bounds[param][0]
                        for param in self.parameter_list
                    ]
                ).reshape(-1, 1),
                (1, batch_size),
            ),
            self.parameter_lookup,
        )

    def disable_gradients(self, parameter_name: str):
        self.params[parameter_name].requires_grad = False

    def enable_gradients(self, parameter_name: str):
        self.params[parameter_name].requires_grad = True
