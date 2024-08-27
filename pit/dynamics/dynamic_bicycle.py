from . import Dynamics

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
    def __init__(self, lf, lr, Iz, mass, Df, Cf, Bf, Dr, Cr, Br, Cm, Cr0, Cr2) -> None:
        super().__init__()
        self.lf = torch.nn.Parameter(torch.tensor(lf, dtype=torch.float32))
        self.lr = torch.nn.Parameter(torch.tensor(lr, dtype=torch.float32))
        self.Iz = torch.nn.Parameter(torch.tensor(Iz, dtype=torch.float32))
        self.mass = torch.nn.Parameter(torch.tensor(mass, dtype=torch.float32))
        self.Df = torch.nn.Parameter(torch.tensor(Df, dtype=torch.float32))
        self.Cf = torch.nn.Parameter(torch.tensor(Cf, dtype=torch.float32))
        self.Bf = torch.nn.Parameter(torch.tensor(Bf, dtype=torch.float32))
        self.Dr = torch.nn.Parameter(torch.tensor(Dr, dtype=torch.float32))
        self.Cr = torch.nn.Parameter(torch.tensor(Cr, dtype=torch.float32))
        self.Br = torch.nn.Parameter(torch.tensor(Br, dtype=torch.float32))
        self.Cm = torch.nn.Parameter(torch.tensor(Cm, dtype=torch.float32))
        self.Cr0 = torch.nn.Parameter(torch.tensor(Cr0, dtype=torch.float32))
        self.Cr2 = torch.nn.Parameter(torch.tensor(Cr2, dtype=torch.float32))

    def calculate_tire_forces(self, states, control_inputs):
        """ Get the tire forces at this point

        Args:
            states (): Shape of (B, 7) or (7)
            control_inputs (): Shape of (B, 2) or (2)
        Returns:
            tire_forces (): Shape of (B, 3) or (3) [Frx, Ffy, Fry]
        """
        batch_mode = True if len(states.shape)==2 else False
        device = self.mass.device
        if batch_mode:
            B = states.shape[0]
            tire_forces = torch.zeros((B, 3), device=device)
            alpha_f = states[:, STEERING_ANGLE] - torch.arctan((states[:, YAW_RATE]*self.lf + states[:, VY])/states[:, VX])
            alpha_r = torch.arctan((states[:, YAW_RATE] * self.lr - states[:, VY])/states[:, VX])
            tire_forces[:, FRX] = self.Cm * control_inputs[:, DRIVE_FORCE] - self.Cr0 - self.Cr2 * states[:, VX]**2.0
            tire_forces[:, FFY] = self.Df * torch.sin(self.Cf * torch.arctan(self.Bf * alpha_f))
            tire_forces[:, FRY] = self.Dr * torch.sin(self.Cr * torch.arctan(self.Br * alpha_r))
        else:
            tire_forces = torch.zeros((3), device=device)
            alpha_f = states[STEERING_ANGLE] - torch.arctan((states[YAW_RATE]*self.lf + states[VY])/states[VX])
            alpha_r = torch.arctan((states[YAW_RATE] * self.lr - states[VY])/states[VX])
            tire_forces[FRX] = self.Cm * control_inputs[DRIVE_FORCE] - self.Cr0 - self.Cr2 * states[VX]**2.0
            tire_forces[FFY] = self.Df * torch.sin(self.Cf * torch.arctan(self.Bf * alpha_f))
            tire_forces[FRY] = self.Dr * torch.sin(self.Cr * torch.arctan(self.Br * alpha_r))
        return tire_forces



    def forward(self, states, control_inputs):
        """ Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 7) or (7)
            control_inputs (): Shape of (B, 2) or (2)
        """
        batch_mode = True if len(states.shape)==2 else False
        
        diff = torch.zeros_like(states)
        tire_forces = self.calculate_tire_forces(states, control_inputs)
        if torch.isnan(tire_forces).any():
            # Handle the case when tire forces return NaN
            # Check for specific parameters causing the blow-up
            if torch.isnan(states).any() or torch.isnan(control_inputs).any():
               print("NaN encountered in states or control_inputs.")
               print("states:", states)
               print("control_inputs:", control_inputs)
            if torch.isnan(self.Cm).any() or torch.isnan(self.Cr0).any() or torch.isnan(self.Cr2).any():
                raise ValueError("NaN encountered in Cm, Cr0, or Cr2.")
            if torch.isnan(self.Df).any() or torch.isnan(self.Cf).any() or torch.isnan(self.Bf).any():
                raise ValueError("NaN encountered in Df, Cf, or Bf.")
            if torch.isnan(self.Dr).any() or torch.isnan(self.Cr).any() or torch.isnan(self.Br).any():
                raise ValueError("NaN encountered in Dr, Cr, or Br.")
            if torch.isnan(self.lf).any() or torch.isnan(self.lr).any() or torch.isnan(self.mass).any():
                raise ValueError("NaN encountered in lf, lr, or mass.")
            if torch.isnan(self.Iz).any():
                raise ValueError("NaN encountered in Iz.")
             
            # Add your desired handling logic here

        if batch_mode:
            diff[:, X] = states[:, VX] * torch.cos(states[:, YAW]) - states[:, VY] * torch.sin(states[:, YAW])
            diff[:, Y] = states[:, VX] * torch.sin(states[:, YAW]) - states[:, VY] * torch.cos(states[:, YAW])
            diff[:, YAW] = states[:, YAW_RATE]
            diff[:, VX] = 1.0 / self.mass * (tire_forces[:, FRX] - tire_forces[:, FFY] * torch.sin(states[:, STEERING_ANGLE]) + states[:, VY] * states[:, YAW_RATE] * self.mass)
            diff[:, VY] = 1.0 / self.mass * (tire_forces[:, FRY] + tire_forces[:, FFY] * torch.cos(states[:, STEERING_ANGLE]) - states[:, VX] * states[:, YAW_RATE] * self.mass)
            diff[:, YAW_RATE] = 1.0 / self.Iz * (tire_forces[:, FFY] * self.lf * torch.cos(states[:, STEERING_ANGLE]) - tire_forces[:, FRY] * self.lr)
            diff[:, STEERING_ANGLE] = control_inputs[:, STEER_SPEED]
        else:
            diff[X] = states[VX] * torch.cos(states[YAW]) - states[VY] * torch.sin(states[YAW])
            diff[Y] = states[VX] * torch.sin(states[YAW]) - states[VY] * torch.cos(states[YAW])
            diff[YAW] = states[YAW_RATE]
            diff[VX] = 1.0 / self.mass * (tire_forces[FRX] - tire_forces[FFY] * torch.sin(states[STEERING_ANGLE]) + states[VY] * states[YAW_RATE] * self.mass)
            diff[VY] = 1.0 / self.mass * (tire_forces[FRY] + tire_forces[FFY] * torch.cos(states[STEERING_ANGLE]) - states[VX] * states[YAW_RATE] * self.mass)
            diff[YAW_RATE] = 1.0 / self.Iz * (tire_forces[FFY] * self.lf * torch.cos(states[STEERING_ANGLE]) - tire_forces[FRY] * self.lr)
            diff[STEERING_ANGLE] = control_inputs[STEER_SPEED]
        return diff