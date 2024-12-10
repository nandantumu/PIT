"""This file contains filtering approaches for batched data."""

import torch


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
        logger(
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
    vel_threshold=0.1,
    time_threshold=0.1,
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
        yaw_filter = torch.ones(batched_data.shape[0], dtype=torch.bool)
    if vel_threshold is not None:
        min_vel_per_item = calculate_min_vel_per_item(batched_data, vel_index=vel_index)
        vel_abs_threshold = select_yaw_rate_threshold(batched_data, yaw_index=yaw_index)
        vel_filter = min_vel_per_item >= vel_abs_threshold
    else:
        vel_filter = torch.ones(batched_data.shape[0], dtype=torch.bool)
    if time_threshold is not None:
        max_delta_time_per_item = calculate_max_delta_time_per_item(batched_delta_times)
        time_filter = max_delta_time_per_item <= time_threshold
    else:
        time_filter = torch.ones(batched_data.shape[0], dtype=torch.bool)
    return yaw_filter & vel_filter & time_filter


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


def calculate_min_vel_per_item(batched_target_states, vel_index=3):
    """Calculate the minimum velocity per item in the batched target states.

    Args:
        batched_target_states (torch.Tensor): The batched target states.
        vel_index (int, optional): The index of the velocity in the target states. Defaults to 3.

    Returns:
        torch.Tensor: The minimum velocity per item in the batched target states.
    """
    return torch.min(batched_target_states[:, :, vel_index], dim=1).values
