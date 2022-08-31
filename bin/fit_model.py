# %%
import torch
from torch import nn
import numpy as np

from pit.dynamics.dynamic_bicycle import DynamicBicycle
from pit.integration import Euler, RK4

import matplotlib.pyplot as plt
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

# input parameters
map_name = 'SaoPaulo'  # Nuerburgring,  SaoPaulo
lap_number = 1  # 1 - faster, 2 - slower
friction = '1-1'  # '1-1' - mu_x = 1.1, '0-7' - mu_x = 0.7

# visualization
with open('dataset_%s_%s_lap%s.json' % (friction, map_name, lap_number), 'r') as f:
    data = json.load(f)

# %%
timestep = 0.02
x = torch.tensor(data['x'])
y = torch.tensor(data['y'])
yaw = torch.tensor(data['yaw'])
vx = torch.tensor(data['vx'])
vy = torch.tensor(data['vy'])
yaw_rate = torch.tensor(data['yaw_rate'])
steer_angle = torch.tensor(data['steer_angle'])
drive_force = torch.tensor(data['drive_force'])
steer_speed = torch.tensor(data['steer_speed'])

# %%
initial_state = torch.tensor([x[0],y[0],yaw[0],vx[0],vy[0],yaw_rate[0],steer_angle[0]]).to(DEVICE)
control_inputs = torch.vstack([drive_force, steer_speed]).T[:600,]
control_inputs = control_inputs.contiguous().to(DEVICE)
target_states = torch.vstack([x, y, yaw, vx, vy, yaw_rate, steer_angle]).T[:600]
target_states = target_states.contiguous().to(DEVICE)

print(f"Inputs size: {control_inputs.shape} | States size: {target_states.shape}")
# %%
params = {
    # axes distances
    'lf': 0.88392,  # distance from spring mass center of gravity to front axle [m]  LENA
    'lr': 1.50876,  # distance from spring mass center of gravity to rear axle [m]  LENB

    # moments of inertia of sprung mass
    'Iz': 1538.853371,  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ

    # masses
    'mass': 1225.887,  # vehicle mass [kg]  MASS

    # Pacejka tire force parameters
    'Df': -0.623359580,  # [rad/m]  DF
    'Cf': 1.0,
    'Bf': 1.0,
    'Dr': -0.209973753,  # [rad/m]  DR
    'Cr': 1.0,
    'Br': 1.0,
    'Cm': 1.0,
    'Cr0': 1.0,
    'Cr2': 1.0,
}
dynamics = DynamicBicycle(**params)
euler_integrator = Euler(dynamics, timestep=timestep)
euler_integrator.to(DEVICE)
rk4_integrator = RK4(dynamics=dynamics, timestep=timestep)
rk4_integrator.to(DEVICE)

euler_output_states = euler_integrator(initial_state, control_inputs)
rk4_output_states = rk4_integrator(initial_state, control_inputs)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10,10))
LEN=10000
ax.plot(target_states[:LEN, 0].cpu().numpy(), target_states[:LEN, 1].cpu().numpy(), label="Target")
ax.plot(euler_output_states[:LEN, 0].detach().cpu().numpy(), euler_output_states[:LEN, 1].detach().cpu().numpy(), label="Euler Prediction")
ax.plot(rk4_output_states[:LEN, 0].detach().cpu().numpy(), rk4_output_states[:LEN, 1].detach().cpu().numpy(), label="RK4 Prediction")
ax.legend()

# %%
X, Y, THETA, V = 0, 1, 2, 3
STEER, ACCEL = 0, 1
EPOCHS = 10
integrator = rk4_integrator
for i in range(EPOCHS):
    optimizer = torch.optim.Adam(integrator.parameters(), lr=.1)
    optimizer.zero_grad()
    output_states = integrator(initial_state, control_inputs)
    loss = torch.nn.functional.l1_loss(output_states, target_states)
    loss.backward()
    optimizer.step()
    for name, param in integrator.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    LEN=10000
    ax.plot(target_states[:LEN, 0].cpu().numpy(), target_states[:LEN, 1].cpu().numpy(), label="Target")
    ax.plot(output_states[:LEN, 0].detach().cpu().numpy(), output_states[:LEN, 1].detach().cpu().numpy(), label="Prediction")
    ax.legend()
    plt.show()
# %%
from viz_net import make_dot
make_dot(integrator)
# %%
