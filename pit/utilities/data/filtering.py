"""This file contains filtering approaches for batched data."""

import torch


def filter_batched_data(
    batched_data, yaw_index=4, vel_index=3, yaw_threshold=None, vel_threshold=None
):
    """Filter the batched data based on the yaw rate and velocity thresholds.

    Args:
        batched_data (torch.Tensor): The batched data.
        yaw_index (int, optional): The index of the yaw rate in the data. Defaults to 4.
        vel_index (int, optional): The index of the velocity in the data. Defaults to 3.
        yaw_threshold (float, optional): The yaw rate threshold. Defaults to None.
        vel_threshold (float, optional): The velocity threshold. Defaults to None.

    Returns:
        torch.Tensor: The filtered batched data.
    """
    filter = create_filter(
        batched_data, yaw_index, vel_index, yaw_threshold, vel_threshold
    )
    return batched_data[filter]


def create_filter(
    batched_data, yaw_index=4, vel_index=3, yaw_threshold=0.8, vel_threshold=0.1
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

    return yaw_filter & vel_filter


def select_yaw_rate_threshold(batched_data, yaw_index=4, yaw_threshold=0.8):
    """This method selects 100 or the top 20% of the yaw rates as the yaw rate threshold."""
    max_yr_per_item = calculate_max_yaw_rate_per_item(batched_data, yaw_index=yaw_index)
    return torch.quantile(max_yr_per_item, yaw_threshold)


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