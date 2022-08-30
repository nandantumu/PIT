# %%
from time import time
import torch
from torch import nn
import numpy as np

from pit.dynamics.kinematic_bicycle import Bicycle
from pit.integration import Euler, RK4

import matplotlib.pyplot as plt
import json

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
theta = torch.tensor(data['yaw'])
v = torch.sqrt(torch.tensor(data['vx'])**2 + torch.tensor(data['vy'])**2)
steer_angle = torch.tensor(data['steer_angle'])
accel = torch.zeros_like(v)
for i in range(1,v.shape[0]):
    accel[i] = v[i] - v[i-1]


# %%
initial_state = torch.tensor([x[0],y[0],theta[0],v[0]])#.unsqueeze(0)
control_inputs = torch.vstack([steer_angle, accel]).T[1:4]
control_inputs = control_inputs.contiguous()#.unsqueeze(0)
target_states = torch.vstack([x, y, theta, v]).T[1:4]
target_states = target_states.contiguous()#.unsqueeze(0)

print(f"Inputs size: {control_inputs.shape} | States size: {target_states.shape}")
# %%
dynamics = Bicycle(4)
integrator = Euler(dynamics, timestep=timestep)

output_states = integrator(initial_state, control_inputs)
# %%
X, Y, THETA, V = 0, 1, 2, 3
STEER, ACCEL = 0, 1
torch.autograd.set_detect_anomaly(True)
EPOCHS = 10
for i in range(EPOCHS):
    optimizer = torch.optim.Adam(integrator.parameters(), lr=0.1)
    optimizer.zero_grad()
    output_states = integrator(initial_state, control_inputs)
    loss = torch.nn.functional.l1_loss(output_states, target_states)
    loss.backward()
    optimizer.step()
    print(dynamics.wb)
# %%
from viz_net import make_dot
make_dot(integrator)
# %%
