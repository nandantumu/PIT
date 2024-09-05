import pytest
import torch
from pit.integration import RK4, Euler
from pit.dynamics.unicycle import Unicycle

def test_time_delta_euler():
    unicycle = Unicycle()
    euler = Euler(unicycle, timestep=0.1)
    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    control_inputs = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    euler_states = euler(initial_state, control_inputs)
    skip_control_inputs = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    time_deltas = torch.tensor([0.1, 0.2])
    euler_states_skip = euler(initial_state, skip_control_inputs, time_deltas)
    assert torch.allclose(euler_states[-1], euler_states_skip[-1])

def test_time_delta_rk4():
    unicycle = Unicycle()
    rk4 = RK4(unicycle, timestep=0.1)
    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    control_inputs = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    rk4_states = rk4(initial_state, control_inputs)
    skip_control_inputs = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    time_deltas = torch.tensor([0.1, 0.2])
    rk4_states_skip = rk4(initial_state, skip_control_inputs, time_deltas)
    assert torch.allclose(rk4_states[-1], rk4_states_skip[-1])