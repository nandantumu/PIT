# %%
import torch
from torch import nn
import numpy as np

from pit.dynamics.kinematic_bicycle import Bicycle
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
theta = torch.tensor(data['yaw'])
v = torch.sqrt(torch.tensor(data['vx'])**2 + torch.tensor(data['vy'])**2)
steer_angle = torch.tensor(data['steer_angle'])
accel = torch.zeros_like(v)
for i in range(1,v.shape[0]):
    accel[i] = v[i] - v[i-1]


# %%
initial_state = torch.tensor([x[0],y[0],theta[0],v[0]]).to(DEVICE)
control_inputs = torch.vstack([steer_angle, accel]).T
control_inputs = control_inputs.contiguous().to(DEVICE)
target_states = torch.vstack([x, y, theta, v]).T
target_states = target_states.contiguous().to(DEVICE)

print(f"Inputs size: {control_inputs.shape} | States size: {target_states.shape}")
# %%
dynamics = Bicycle(4.2)
euler_integrator = Euler(dynamics, timestep=timestep)
euler_integrator.to(DEVICE)
rk4_integrator = RK4(dynamics=dynamics, timestep=timestep)
rk4_integrator.to(DEVICE)

euler_output_states = euler_integrator(initial_state, control_inputs)
rk4_output_states = rk4_integrator(initial_state, control_inputs)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10,10))
LEN=10000
#ax.plot(target_states[:LEN, 0].cpu().numpy(), target_states[:LEN, 1].cpu().numpy(), label="Target")
ax.plot(euler_output_states[:LEN, 0].detach().cpu().numpy(), euler_output_states[:LEN, 1].detach().cpu().numpy(), label="Euler Prediction")
ax.plot(rk4_output_states[:LEN, 0].detach().cpu().numpy(), rk4_output_states[:LEN, 1].detach().cpu().numpy(), label="RK4 Prediction")
ax.legend()

# %%
X, Y, THETA, V = 0, 1, 2, 3
STEER, ACCEL = 0, 1
EPOCHS = 10
dynamics = Bicycle(4.2)
integrator = Euler(dynamics=dynamics, timestep=timestep)
for i in range(EPOCHS):
    optimizer = torch.optim.Adam(integrator.parameters(), lr=1)
    optimizer.zero_grad()
    output_states = integrator(initial_state, control_inputs)
    loss = torch.nn.functional.l1_loss(output_states, target_states)
    loss.backward()
    optimizer.step()
    print(dynamics.wb)
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
