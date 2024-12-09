import numpy as np
import torch


def create_amz_states_and_controls(data_dict):
    initial_state = torch.tensor(
        [
            data_dict["x"][0],
            data_dict["y"][0],
            data_dict["yaw"][0],
            data_dict["v_x"][0],
            data_dict["v_y"][0],
            data_dict["yaw_rate"][0],
            data_dict["steering_angle"][0],
        ]
    )
    control_inputs = np.array(
        [data_dict["acceleration"], data_dict["steering_velocity"]], dtype=np.float64
    ).T
    control_inputs = torch.tensor(control_inputs[:-1])
    output_states = torch.tensor(
        np.array(
            [
                data_dict["x"],
                data_dict["y"],
                data_dict["yaw"],
                data_dict["v_x"],
                data_dict["v_y"],
                data_dict["yaw_rate"],
                data_dict["steering_angle"],
            ],
            dtype=np.float64,
        )
    ).T
    target_states = output_states[1:]
    delta_times = data_dict["dt"]
    return (initial_state, control_inputs, output_states, target_states, delta_times)


def create_single_track_states_and_controls(data_dict):
    initial_state = torch.tensor(
        [
            data_dict["x"][0],
            data_dict["y"][0],
            data_dict["velocity"][0],
            data_dict["yaw"][0],
            data_dict["yaw_rate"][0],
            data_dict["slip_angle"][0],
        ]
    )

    control_inputs = torch.vstack(
        [data_dict["steering_angle"], data_dict["acceleration"]]
    ).T
    control_inputs = torch.tensor(control_inputs[:-1])
    output_states = torch.vstack(
        [
            data_dict["x"],
            data_dict["y"],
            data_dict["velocity"],
            data_dict["yaw"],
            data_dict["yaw_rate"],
            data_dict["slip_angle"],
        ]
    ).T
    target_states = output_states[1:]
    delta_times = data_dict["dt"]
    return (initial_state, control_inputs, output_states, target_states, delta_times)


def create_single_track_drift_states_and_controls(data_dict):
    initial_state = torch.tensor(
        [
            data_dict["x"][0],
            data_dict["y"][0],
            data_dict["steering_angle"][0],
            data_dict["velocity"][0],
            data_dict["yaw"][0],
            data_dict["yaw_rate"][0],
            data_dict["slip_angle"][0],
            data_dict["omega_f"][0],
            data_dict["omega_r"][0],
        ]
    )
    control_inputs = np.array(
        [data_dict["steering_velocity"], data_dict["acceleration"]], dtype=np.float64
    ).T
    control_inputs = torch.tensor(control_inputs[:-1])
    output_states = torch.tensor(
        np.array(
            [
                data_dict["x"],
                data_dict["y"],
                data_dict["steering_angle"],
                data_dict["velocity"],
                data_dict["yaw"],
                data_dict["yaw_rate"],
                data_dict["slip_angle"],
                data_dict["omega_f"],
                data_dict["omega_r"],
            ],
            dtype=np.float64,
        )
    ).T
    target_states = output_states[1:]
    delta_times = data_dict["dt"]
    return (initial_state, control_inputs, output_states, target_states, delta_times)
