from . import Dynamics
from ..parameters import PointParameterGroup, CovariantNormalParameterGroup, NormalParameterGroup
from ..parameters.definitions import ParameterSample

import torch
from torch import nn

X, Y, YAW, VX, VY, YAW_RATE, STEERING_ANGLE = 0, 1, 2, 3, 4, 5, 6
DRIVE_FORCE, STEER_SPEED = 0, 1
FRX, FFY, FRY = 0, 1, 2

class DynamicBicycle(Dynamics, nn.Module):
    """
    This is a dynamic bicycle model 
    From AMZ Driverless: The Full Autonomous Racing System
    Model reference point: CoG
    Longitudinal drive-train forces act on the center of gravity
    State Variable [x, y, yaw, vx, vy, yaw rate, steering angle]
    Control Inputs [drive force, steering speed]
    """
    def __init__(self, lf, lr, Iz, m, Df, Cf, Bf, Dr, Cr, Br, Cm, Cr0, Cr2, **kwargs) -> None:
        super().__init__()
        self.parameter_list = ['lf', 'lr', 'Iz', 'm', 'Df', 'Cf', 'Bf', 'Dr', 'Cr', 'Br', 'Cm', 'Cr0', 'Cr2']
        self.initial_values = {
            'lf': lf,
            'lr': lr,
            'Iz': Iz,
            'm': m,
            'Df': Df,
            'Cf': Cf,
            'Bf': Bf,
            'Dr': Dr,
            'Cr': Cr,
            'Br': Br,
            'Cm': Cm,
            'Cr0': Cr0,
            'Cr2': Cr2
        }
        # if param_type == 'point':
        #     self.params = PointParameterGroup(self.param_names, self.initial_values)
        # elif param_type == 'normal':
        #     self.params = NormalParameterGroup(self.param_names, self.initial_values)
        # elif param_type == 'covariant':
        #     # raise FutureWarning("CovariantNormalParameterGroup is not implemented yet")
        #     self.params = CovariantNormalParameterGroup(self.param_names, self.initial_values)

        # self.lf = torch.nn.Parameter(torch.tensor(lf, dtype=torch.float32))
        # self.lr = torch.nn.Parameter(torch.tensor(lr, dtype=torch.float32))
        # self.Iz = torch.nn.Parameter(torch.tensor(Iz, dtype=torch.float32))
        # self.mass = torch.nn.Parameter(torch.tensor(mass, dtype=torch.float32))
        # self.Df = torch.nn.Parameter(torch.tensor(Df, dtype=torch.float32))
        # self.Cf = torch.nn.Parameter(torch.tensor(Cf, dtype=torch.float32))
        # self.Bf = torch.nn.Parameter(torch.tensor(Bf, dtype=torch.float32))
        # self.Dr = torch.nn.Parameter(torch.tensor(Dr, dtype=torch.float32))
        # self.Cr = torch.nn.Parameter(torch.tensor(Cr, dtype=torch.float32))
        # self.Br = torch.nn.Parameter(torch.tensor(Br, dtype=torch.float32))
        # self.Cm = torch.nn.Parameter(torch.tensor(Cm, dtype=torch.float32))
        # self.Cr0 = torch.nn.Parameter(torch.tensor(Cr0, dtype=torch.float32))
        # self.Cr2 = torch.nn.Parameter(torch.tensor(Cr2, dtype=torch.float32))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # self.params.to(*args, **kwargs)

    def calculate_tire_forces(self, states, control_inputs, params: ParameterSample):
        """ Get the tire forces at this point

        Args:
            states (): Shape of (B, 7) or (7)
            control_inputs (): Shape of (B, 2) or (2)
        Returns:
            tire_forces (): Shape of (B, 3) or (3) [Frx, Ffy, Fry]
        """
        batch_mode = True if len(states.shape)==2 else False
        device = params['lf'].device
        if batch_mode:
            B = states.shape[0]
            tire_forces = torch.zeros((B, 3), device=device)
            alpha_f = states[:, STEERING_ANGLE] - torch.arctan((states[:, YAW_RATE]*params['lf'] + states[:, VY])/states[:, VX])
            alpha_r = torch.arctan((states[:, YAW_RATE] * params['lr'] - states[:, VY])/states[:, VX])
            tire_forces[:, FRX] = params['Cm'] * control_inputs[:, DRIVE_FORCE] - params['Cr0'] - params['Cr2'] * states[:, VX]**2.0
            tire_forces[:, FFY] = params['Df'] * torch.sin(params['Cf'] * torch.arctan(params['Bf'] * alpha_f))
            tire_forces[:, FRY] = params['Dr'] * torch.sin(params['Cr'] * torch.arctan(params['Br'] * alpha_r))
        else:
            tire_forces = torch.zeros((3), device=device)
            alpha_f = states[STEERING_ANGLE] - torch.arctan((states[YAW_RATE]*params['lf'] + states[VY])/states[VX])
            alpha_r = torch.arctan((states[YAW_RATE] * params['lr'] - states[VY])/states[VX])
            tire_forces[FRX] = params['Cm'] * control_inputs[DRIVE_FORCE] - params['Cr0'] - params['Cr2'] * states[VX]**2.0
            tire_forces[FFY] = params['Df'] * torch.sin(params['Cf'] * torch.arctan(params['Bf'] * alpha_f))
            tire_forces[FRY] = params['Dr'] * torch.sin(params['Cr'] * torch.arctan(params['Br'] * alpha_r))
        return tire_forces


    def forward(self, states, control_inputs, params: ParameterSample):
        """ Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 7) or (7)
            control_inputs (): Shape of (B, 2) or (2)
            params (): Shape of (B, 13) or (13)
        """
        # mass, lf, lr, Iz = params[:, 3], params[:, 0], params[:, 1], params[:, 2]
        batch_mode = True if len(states.shape)==2 else False
        
        diff = torch.zeros_like(states)
        tire_forces = self.calculate_tire_forces(states, control_inputs, params)
        if batch_mode:
            diff[:, X] = states[:, VX] * torch.cos(states[:, YAW]) - states[:, VY] * torch.sin(states[:, YAW])
            diff[:, Y] = states[:, VX] * torch.sin(states[:, YAW]) - states[:, VY] * torch.cos(states[:, YAW])
            diff[:, YAW] = states[:, YAW_RATE]
            diff[:, VX] = 1.0 / params['m'] * (tire_forces[:, FRX] - tire_forces[:, FFY] * torch.sin(states[:, STEERING_ANGLE]) + states[:, VY] * states[:, YAW_RATE] * params['m'])
            diff[:, VY] = 1.0 / params['m'] * (tire_forces[:, FRY] + tire_forces[:, FFY] * torch.cos(states[:, STEERING_ANGLE]) - states[:, VX] * states[:, YAW_RATE] * params['m'])
            diff[:, YAW_RATE] = 1.0 / params['Iz'] * (tire_forces[:, FFY] * params['lf'] * torch.cos(states[:, STEERING_ANGLE]) - tire_forces[:, FRY] * params['lr'])
            diff[:, STEERING_ANGLE] = control_inputs[:, STEER_SPEED]
        else:
            diff[X] = states[VX] * torch.cos(states[YAW]) - states[VY] * torch.sin(states[YAW])
            diff[Y] = states[VX] * torch.sin(states[YAW]) - states[VY] * torch.cos(states[YAW])
            diff[YAW] = states[YAW_RATE]
            diff[VX] = 1.0 / params['m'] * (tire_forces[FRX] - tire_forces[FFY] * torch.sin(states[STEERING_ANGLE]) + states[VY] * states[YAW_RATE] * params['m'])
            diff[VY] = 1.0 / params['m'] * (tire_forces[FRY] + tire_forces[FFY] * torch.cos(states[STEERING_ANGLE]) - states[VX] * states[YAW_RATE] * params['m'])
            diff[YAW_RATE] = 1.0 / params['Iz'] * (tire_forces[FFY] * params['lf'] * torch.cos(states[STEERING_ANGLE]) - tire_forces[FRY] * params['lr'])
            diff[STEERING_ANGLE] = control_inputs[STEER_SPEED]
        return diff