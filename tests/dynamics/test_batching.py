from __future__ import annotations

import torch

from pit.dynamics.dynamic_bicycle import DynamicBicycle
from pit.dynamics.kinematic_bicycle import Bicycle
from pit.dynamics.unicycle import Unicycle
from pit.parameters.definitions import ParameterSample


def _make_parameter_sample(values):
    names = ['lf', 'lr', 'Iz', 'm', 'Df', 'Cf', 'Bf', 'Dr', 'Cr', 'Br', 'Cm', 'Cr0', 'Cr2']
    lookup = {name: idx for idx, name in enumerate(names)}
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return ParameterSample(tensor, lookup)


def test_dynamic_bicycle_single_matches_batch():
    model = DynamicBicycle(
        lf=1.2,
        lr=1.3,
        Iz=1400.0,
        m=1500.0,
        Df=1.0,
        Cf=1.2,
        Bf=10.0,
        Dr=1.0,
        Cr=1.3,
        Br=11.0,
        Cm=0.5,
        Cr0=0.1,
        Cr2=0.01,
    )

    state = torch.tensor([0.0, 0.0, 0.2, 5.0, 0.3, 0.05, 0.02], dtype=torch.float32)
    control = torch.tensor([0.4, 0.01], dtype=torch.float32)
    params_single = _make_parameter_sample(
        [1.2, 1.3, 1400.0, 1500.0, 1.0, 1.2, 10.0, 1.0, 1.3, 11.0, 0.5, 0.1, 0.01]
    )
    params_batch = _make_parameter_sample(
        torch.tensor(
            [
                [1.2, 1.3, 1400.0, 1500.0, 1.0, 1.2, 10.0, 1.0, 1.3, 11.0, 0.5, 0.1, 0.01],
            ]
        ).T
    )

    diff_single = model.forward(state, control, params_single)
    diff_batch = model.forward(state.unsqueeze(0), control.unsqueeze(0), params_batch)
    tire_single = model.calculate_tire_forces(state, control, params_single)
    tire_batch = model.calculate_tire_forces(state.unsqueeze(0), control.unsqueeze(0), params_batch)

    torch.testing.assert_close(diff_batch.squeeze(0), diff_single)
    torch.testing.assert_close(tire_batch.squeeze(0), tire_single)


def test_dynamic_bicycle_multi_batch_matches_single_calls():
    model = DynamicBicycle(
        lf=1.2,
        lr=1.3,
        Iz=1400.0,
        m=1500.0,
        Df=1.0,
        Cf=1.2,
        Bf=10.0,
        Dr=1.0,
        Cr=1.3,
        Br=11.0,
        Cm=0.5,
        Cr0=0.1,
        Cr2=0.01,
    )

    states = torch.stack(
        [
            torch.tensor([0.0, 0.0, 0.1, 5.0, 0.2, 0.05, 0.02], dtype=torch.float32),
            torch.tensor([1.0, -0.5, 0.3, 7.0, -0.1, 0.1, -0.05], dtype=torch.float32),
            torch.tensor([-2.0, 0.4, -0.2, 3.5, 0.15, -0.02, 0.03], dtype=torch.float32),
        ]
    )
    controls = torch.stack(
        [
            torch.tensor([0.4, 0.01], dtype=torch.float32),
            torch.tensor([-0.2, 0.05], dtype=torch.float32),
            torch.tensor([0.1, -0.03], dtype=torch.float32),
        ]
    )
    base_params = torch.tensor([1.2, 1.3, 1400.0, 1500.0, 1.0, 1.2, 10.0, 1.0, 1.3, 11.0, 0.5, 0.1, 0.01], dtype=torch.float32)
    params_matrix = torch.stack(
        [
            base_params,
            base_params * 1.01,
            base_params * 0.99,
        ]
    ).T
    params_batch = _make_parameter_sample(params_matrix)

    batch_result = model.forward(states, controls, params_batch)
    batch_tire = model.calculate_tire_forces(states, controls, params_batch)

    singles = []
    tires = []
    for idx in range(states.shape[0]):
        params_single = _make_parameter_sample(params_matrix[:, idx])
        singles.append(model.forward(states[idx], controls[idx], params_single))
        tires.append(model.calculate_tire_forces(states[idx], controls[idx], params_single))

    torch.testing.assert_close(batch_result, torch.stack(singles))
    torch.testing.assert_close(batch_tire, torch.stack(tires))


def test_kinematic_bicycle_batching_matches_single():
    model = Bicycle(wheelbase=2.5)
    state = torch.tensor([0.0, 0.0, 0.2, 5.0], dtype=torch.float32)
    control = torch.tensor([0.1, 0.2], dtype=torch.float32)

    diff_single = model.forward(state, control)
    diff_batch = model.forward(state.unsqueeze(0), control.unsqueeze(0))

    torch.testing.assert_close(diff_batch.squeeze(0), diff_single)

    states = torch.stack(
        [
            torch.tensor([0.0, 0.0, 0.1, 5.0], dtype=torch.float32),
            torch.tensor([1.0, -0.5, 0.3, 7.0], dtype=torch.float32),
        ]
    )
    controls = torch.stack(
        [
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            torch.tensor([-0.2, 0.5], dtype=torch.float32),
        ]
    )

    batch_result = model.forward(states, controls)
    singles = [model.forward(states[i], controls[i]) for i in range(states.shape[0])]
    torch.testing.assert_close(batch_result, torch.stack(singles))


def test_unicycle_batching_matches_single():
    model = Unicycle()
    state = torch.tensor([0.0, 0.0, 0.2, 5.0], dtype=torch.float32)
    control = torch.tensor([0.1, 0.2], dtype=torch.float32)

    diff_single = model.forward(state, control, params=None)
    diff_batch = model.forward(state.unsqueeze(0), control.unsqueeze(0), params=None)
    torch.testing.assert_close(diff_batch.squeeze(0), diff_single)

    states = torch.stack(
        [
            torch.tensor([0.0, 0.0, 0.1, 5.0], dtype=torch.float32),
            torch.tensor([1.0, -0.5, 0.3, 7.0], dtype=torch.float32),
        ]
    )
    controls = torch.stack(
        [
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            torch.tensor([-0.2, 0.5], dtype=torch.float32),
        ]
    )

    batch_result = model.forward(states, controls, params=None)
    singles = [model.forward(states[i], controls[i], params=None) for i in range(states.shape[0])]
    torch.testing.assert_close(batch_result, torch.stack(singles))
