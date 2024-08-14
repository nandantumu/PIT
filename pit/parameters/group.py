# import torch
# from torch import nn
# from torch.nn import functional as F
# from .point import PointParameter
# from .distribution import NormalParameter

# class ParameterGroup(nn.Module):
#     def __init__(self, parameter_list: list, initial_value: dict=None, type: str='point'):
#         super().__init__()
#         self.params = nn.ParameterDict()
#         if type == 'point':
#             self.ptype = PointParameter
#         elif type == 'normal':
#             self.ptype = NormalParameter

#         for param in parameter_list:
#             self.params[param] = self.ptype(initial_value[param] if initial_value else None)
        
#     def __getitem__(self, key):
#         return self.params[key]() 
    
#     def __setitem__(self, key, value):
#         self.params[key] = self.ptype(value)

#     def forward(self):
#         return self.params