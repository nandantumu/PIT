import torch
from ..loss import yaw_normalized_loss, yaw_normalized_loss_per_item
from ..data.filtering import create_filter, filter_batched_data


def find_near_optimal_mu(
    batched_initial_states,
    batched_control_inputs,
    batched_delta_times,
    batched_target_states,
    integrator,
    range=(0.1, 1.0),
    num_samples=100,
):
    """Find the near-optimal mu for the given batched data.

    Args:
        batched_initial_states (torch.Tensor): The initial states of the batch.
        batched_control_inputs (torch.Tensor): The control inputs for the batch.
        batched_delta_times (torch.Tensor): The time deltas for the batch.
        batched_target_states (torch.Tensor): The target states for the batch.
        integrator (Integrator): The integrator to use for processing the batched data.
        range (tuple, optional): The range of mu values to sample. Defaults to (0.1, 1.0).
        num_samples (int, optional): The number of mu samples to evaluate. Defaults to 100.

    Returns:
        float: The near-optimal mu value calculated based on the given batched data and integrator.
    """
    # Filter the batched data
    filter = create_filter(batched_target_states)
    filtered_batched_initial_states = batched_initial_states[filter]
    filtered_batched_control_inputs = batched_control_inputs[filter]
    filtered_batched_delta_times = batched_delta_times[filter]
    filtered_batched_target_states = batched_target_states[filter]

    mu_values = torch.linspace(range[0], range[1], num_samples)
    losses = torch.zeros_like(mu_values)
    for i, mu in enumerate(mu_values):
        integrator.model_params.params["mu"] = torch.tensor(mu)
        with torch.no_grad():
            filtered_batched_output_states = integrator(
                filtered_batched_initial_states,
                filtered_batched_control_inputs,
                filtered_batched_delta_times,
            )
            loss = yaw_normalized_loss(
                filtered_batched_output_states, filtered_batched_target_states
            )
        losses[i] = loss
    min_loss, min_index = torch.min(losses, 0)
    return mu_values[min_index.item()]
