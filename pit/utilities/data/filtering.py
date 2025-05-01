"""This file contains filtering approaches for batched data."""

import torch
import numpy as np


def filter_batched_data(
    batched_initial_states: torch.Tensor,
    batched_control_inputs: torch.Tensor,
    batched_delta_times: torch.Tensor,
    batched_target_states: torch.Tensor,
    logger=None,
) -> tuple:
    """Filter the batched data based on the target states.

    Args:
        batched_initial_states (torch.Tensor): The initial states of the batch.
        batched_control_inputs (torch.Tensor): The control inputs for the batch.
        batched_delta_times (torch.Tensor): The time deltas for the batch.
        batched_target_states (torch.Tensor): The target states for the batch.

    Returns:
        tuple: Filtered batched initial states, control inputs, delta times, and target states.
    """
    filter = create_filter(batched_target_states, batched_delta_times)
    filtered_batched_initial_states = batched_initial_states[filter]
    filtered_batched_control_inputs = batched_control_inputs[filter]
    filtered_batched_delta_times = batched_delta_times[filter]
    filtered_batched_target_states = batched_target_states[filter]

    if logger is not None:
        # Log the number of items filtered
        num_items = batched_initial_states.shape[0]
        num_filtered_items = num_items - filtered_batched_initial_states.shape[0]
        logger.info(
            f"Filtered {num_filtered_items} items out of {num_items}. There are {filtered_batched_initial_states.shape[0]} items remaining."
        )

    return (
        filtered_batched_initial_states,
        filtered_batched_control_inputs,
        filtered_batched_delta_times,
        filtered_batched_target_states,
    )


def create_filter(
    batched_data,
    batched_delta_times,
    yaw_index=4,
    vel_index=3,
    yaw_threshold=None,
    vel_threshold=0.5,
    time_threshold=0.1,
    slip_angle_threshold=(np.pi / 180) * 30,
):
    """Create a filter based on the yaw rate and velocity thresholds.

    Args:
        batched_data (torch.Tensor): The batched data.
        yaw_index (int): The index of the yaw rate in the data.
        vel_index (int): The index of the velocity in the data.
        yaw_threshold (float): The yaw rate threshold.
        vel_threshold (float): The velocity threshold.


    Returns:
        torch.Tensor: The filter for the batched data.
    """
    if yaw_threshold is not None:
        max_yr_per_item = calculate_max_yaw_rate_per_item(
            batched_data, yaw_index=yaw_index
        )
        yaw_filter = max_yr_per_item <= yaw_threshold
    else:
        yaw_filter = torch.ones(
            batched_data.shape[0], dtype=torch.bool, device=batched_data.device
        )
    if vel_threshold is not None:
        min_vel_per_item = calculate_min_vel_per_item(batched_data, vel_index=vel_index)
        vel_abs_threshold = select_yaw_rate_threshold(batched_data, yaw_index=yaw_index)
        vel_filter = min_vel_per_item >= vel_abs_threshold
    else:
        vel_filter = torch.ones(
            batched_data.shape[0], dtype=torch.bool, device=batched_data.device
        )
    if time_threshold is not None:
        max_delta_time_per_item = calculate_max_delta_time_per_item(batched_delta_times)
        time_filter = max_delta_time_per_item <= time_threshold
    else:
        time_filter = torch.ones(
            batched_data.shape[0], dtype=torch.bool, device=batched_data.device
        )
    if slip_angle_threshold is not None:
        min_slip_angle_per_item = calculate_min_slip_angle_per_item(
            batched_data, slip_angle_index=5
        )
        max_slip_angle_per_item = calculate_max_slip_angle_per_item(
            batched_data, slip_angle_index=5
        )
        slip_angle_filter = (min_slip_angle_per_item >= -slip_angle_threshold) & (
            max_slip_angle_per_item <= slip_angle_threshold
        )
    # Ensure that the derivative of the yaw rate is not 0
    # yr_deriv = (batched_data[:, 1:, 4] - batched_data[:, :-1, 4]) / batched_delta_times[
    #     :, 1:
    # ]
    # mean_yr_deriv = torch.mean(torch.abs(yr_deriv), dim=1)
    # yr_deriv_filter = mean_yr_deriv >= 0.05
    # Ensure that the yaw_rate is greater than 0
    # yaw_rate_filter = torch.min(torch.abs(batched_data[:, :, 4]), dim=1).values > 0.2
    return (
        yaw_filter & vel_filter & time_filter & slip_angle_filter
        # & yr_deriv_filter
        # & yaw_rate_filter
    )


def select_yaw_rate_threshold(batched_data, yaw_index=4, yaw_threshold=0.5):
    """This method selects 100 or the top 20% of the yaw rates as the yaw rate threshold."""
    max_yr_per_item = calculate_max_yaw_rate_per_item(batched_data, yaw_index=yaw_index)
    return torch.quantile(max_yr_per_item, yaw_threshold)


def calculate_max_delta_time_per_item(batched_delta_times):
    """Calculate the maximum delta time per item in the batched delta times.

    Args:
        batched_delta_times (torch.Tensor): The batched delta times.

    Returns:
        torch.Tensor: The maximum delta time per item in the batched delta times.
    """
    return torch.max(batched_delta_times, dim=1).values


def calculate_max_yaw_rate_per_item(batched_target_states, yaw_index=4):
    """Calculate the maximum yaw rate per item in the batched target states.

    Args:
        batched_target_states (torch.Tensor): The batched target states.
        yaw_index (int, optional): The index of the yaw rate in the target states. Defaults to 4.

    Returns:
        torch.Tensor: The maximum yaw rate per item in the batched target states.
    """
    return torch.max(torch.abs(batched_target_states[:, :, yaw_index]), dim=1).values


def calculate_min_slip_angle_per_item(batched_target_states, slip_angle_index=5):
    """Calculate the minimum slip angle per item in the batched target states.

    Args:
        batched_target_states (torch.Tensor): The batched target states.
        slip_angle_index (int, optional): The index of the slip angle in the target states. Defaults to 5.

    Returns:
        torch.Tensor: The minimum slip angle per item in the batched target states.
    """
    return torch.min(batched_target_states[:, :, slip_angle_index], dim=1).values


def calculate_max_slip_angle_per_item(batched_target_states, slip_angle_index=5):
    """Calculate the maximum slip angle per item in the batched target states.

    Args:
        batched_target_states (torch.Tensor): The batched target states.
        slip_angle_index (int, optional): The index of the slip angle in the target states. Defaults to 5.

    Returns:
        torch.Tensor: The maximum slip angle per item in the batched target states.
    """
    return torch.max(batched_target_states[:, :, slip_angle_index], dim=1).values


def calculate_min_vel_per_item(batched_target_states, vel_index=3):
    """Calculate the minimum velocity per item in the batched target states.

    Args:
        batched_target_states (torch.Tensor): The batched target states.
        vel_index (int, optional): The index of the velocity in the target states. Defaults to 3.

    Returns:
        torch.Tensor: The minimum velocity per item in the batched target states.
    """
    return torch.min(batched_target_states[:, :, vel_index], dim=1).values

def calculate_max_distance_between_items(
    batched_target_states, x_index=0, y_index=1
):
    """Calculate the maximum distance between items in the batched target states.

    Args:
        batched_target_states (torch.Tensor): The batched target states.
        x_index (int, optional): The index of the x-coordinate in the target states. Defaults to 0.
        y_index (int, optional): The index of the y-coordinate in the target states. Defaults to 1.

    Returns:
        torch.Tensor: The maximum distance between items in the batched target states.
    """
    # Calculate the distance between each pair of items
    distances = torch.cdist(
        batched_target_states[:, :, [x_index, y_index]],
        batched_target_states[:, :, [x_index, y_index]],
    )
    # Get the maximum distance for each item
    max_distances = torch.max(distances, dim=1).values
    return max_distances