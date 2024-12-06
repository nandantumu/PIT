import matplotlib.pyplot as plt
import numpy as np
import torch


def import_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    return_dict = {}
    return_dict["time"] = torch.tensor(
        [item["time"] for item in data], dtype=torch.float64
    )
    return_dict["dt"] = return_dict["time"][1:] - return_dict["time"][:-1]
    # Delete the last element of time, and trim the last element of all other tensors
    return_dict["time"] = return_dict["time"][:-1]
    return_dict["x"] = torch.tensor([item["x"] for item in data], dtype=torch.float64)[
        :-1
    ]
    return_dict["y"] = torch.tensor([item["y"] for item in data], dtype=torch.float64)[
        :-1
    ]
    return_dict["steering_angle"] = torch.tensor(
        [item["steering_angle"] for item in data], dtype=torch.float64
    )[:-1]
    return_dict["velocity"] = torch.tensor(
        [item["velocity"] for item in data], dtype=torch.float64
    )[:-1]
    try:
        return_dict["v_x"] = torch.tensor(
            [item["v_x"] for item in data], dtype=torch.float64
        )[:-1]
        return_dict["v_y"] = torch.tensor(
            [item["v_y"] for item in data], dtype=torch.float64
        )[:-1]
    except KeyError:
        # v_x = None
        # v_y = None
        pass
    return_dict["yaw"] = torch.tensor(
        [item["yaw"] for item in data], dtype=torch.float64
    )[:-1]
    return_dict["yaw_rate"] = torch.tensor(
        [item["yaw_rate"] for item in data], dtype=torch.float64
    )[:-1]
    return_dict["slip_angle"] = torch.tensor(
        [item["slip_angle"] for item in data], dtype=torch.float64
    )[:-1]
    try:
        return_dict["omega_f"] = torch.tensor(
            [item["omega_f"] for item in data], dtype=torch.float64
        )[:-1]
        return_dict["omega_r"] = torch.tensor(
            [item["omega_r"] for item in data], dtype=torch.float64
        )[:-1]
    except KeyError:
        # omega_f = None
        # omega_r = None
        pass

    return_dict["steering_velocity"] = torch.tensor(
        [item["steering_velocity"] for item in data], dtype=torch.float64
    )[:-1]
    return_dict["acceleration"] = torch.tensor(
        [item["acceleration"] for item in data], dtype=torch.float64
    )[:-1]

    return return_dict


def trim_data_length(data_dict, start_offset=None, end_offset=None, step_size=None):
    return {
        key: value[slice(start_offset, end_offset, step_size)]
        for key, value in data_dict.items()
    }


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
    control_inputs = np.array(
        [data_dict["steering_angle"], data_dict["acceleration"]], dtype=np.float64
    ).T
    control_inputs = torch.tensor(control_inputs[:-1])
    output_states = torch.tensor(
        np.array(
            [
                data_dict["x"],
                data_dict["y"],
                data_dict["velocity"],
                data_dict["yaw"],
                data_dict["yaw_rate"],
                data_dict["slip_angle"],
            ],
            dtype=np.float64,
        )
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


def create_batched_track_states_and_controls(
    initial_state,
    control_inputs,
    target_states,
    output_states,
    delta_times,
    step_size,
    ticks_in_step,
):
    state_dims = initial_state.shape[0]
    input_dims = control_inputs.shape[-1]
    total_steps = (control_inputs.shape[0] - ticks_in_step - 1) // step_size

    batched_inital_states = torch.zeros((total_steps, state_dims))
    batched_control_inputs = torch.zeros((total_steps, ticks_in_step, input_dims))
    batched_target_states = torch.zeros((total_steps, ticks_in_step, state_dims))
    batched_delta_times = torch.zeros((total_steps, ticks_in_step))

    for step in range(total_steps):
        start_index = step * step_size
        end_index = start_index + ticks_in_step
        batched_inital_states[step] = output_states[start_index]
        batched_control_inputs[step] = control_inputs[start_index:end_index]
        batched_target_states[step] = target_states[start_index + 1 : end_index + 1]
        batched_delta_times[step] = delta_times[start_index:end_index]

    return (
        batched_inital_states,
        batched_control_inputs,
        batched_target_states,
        batched_delta_times,
    )
