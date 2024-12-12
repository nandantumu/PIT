import torch
from ..loss import yaw_normalized_loss, yaw_normalized_loss_per_item
from ..data.filtering import create_filter, filter_batched_data
from tqdm.auto import trange


def find_near_optimal_mu(
    batched_initial_states,
    batched_control_inputs,
    batched_delta_times,
    batched_target_states,
    integrator,
    range=(0.1, 1.0),
    num_samples=100,
    logger=None,
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
        float: The optimal mu value calculated based on the given batched data and integrator.
        tuple: A tuple containing the mu values and losses for each mu value.
    """
    # Filter the batched data
    (
        filtered_batched_initial_states,
        filtered_batched_control_inputs,
        filtered_batched_delta_times,
        filtered_batched_target_states,
    ) = filter_batched_data(
        batched_initial_states,
        batched_control_inputs,
        batched_delta_times,
        batched_target_states,
        logger=logger,
    )

    return mu_search(
        filtered_batched_initial_states,
        filtered_batched_control_inputs,
        filtered_batched_delta_times,
        filtered_batched_target_states,
        integrator,
        range=range,
        num_samples=num_samples,
        logger=logger,
    )


def mu_search(
    batched_initial_states,
    batched_control_inputs,
    batched_delta_times,
    batched_target_states,
    integrator,
    range=(0.1, 1.0),
    num_samples=100,
    logger=None,
):
    """
    Perform a search for the optimal mu value for the given batched data.

    Args:
        batched_initial_states (torch.Tensor): The initial states of the batch.
        batched_control_inputs (torch.Tensor): The control inputs for the batch.
        batched_delta_times (torch.Tensor): The time deltas for the batch.
        batched_target_states (torch.Tensor): The target states for the batch.
        integrator (Integrator): The integrator to use for processing the batched data.
        range (tuple, optional): The range of mu values to sample. Defaults to (0.1, 1.0).
        num_samples (int, optional): The number of mu samples to evaluate. Defaults to 100.

    Returns:
        float: The optimal mu value calculated based on the given batched data and integrator.
        tuple: A tuple containing the mu values and losses for each mu value.
    """
    mu_values = torch.linspace(range[0], range[1], num_samples)
    losses = torch.zeros_like(mu_values)
    for i, mu in enumerate(mu_values):
        integrator.model_params.params["mu"] = torch.tensor(mu)
        with torch.no_grad():
            batched_output_states = integrator(
                batched_initial_states,
                batched_control_inputs,
                batched_delta_times,
            )
            loss = yaw_normalized_loss(batched_output_states, batched_target_states)
        losses[i] = loss
    min_loss, min_index = torch.min(losses, 0)
    return mu_values[min_index.item()], (mu_values, losses)


def gradient_search_for_mu(
    batched_initial_states,
    batched_control_inputs,
    batched_delta_times,
    batched_target_states,
    integrator,
    range=(0.1, 1.0),
    initial_mu_guess=0.1,
    num_samples=100,
    batch_size=1024,
    lr=0.1,
    epochs: int = 100,
    logger=None,
):
    """
    Perform gradient-based search to estimate the friction coefficient (mu) for a given integrator model.

    Args:
        batched_initial_states (torch.Tensor): Batched initial states of the system.
        batched_control_inputs (torch.Tensor): Batched control inputs applied to the system.
        batched_delta_times (torch.Tensor): Batched time intervals between states.
        batched_target_states (torch.Tensor): Batched target states to be achieved.
        integrator (Integrator): The integrator model used for state prediction.
        range (tuple, optional): The range of mu values to search within. Defaults to (0.1, 1.0).
        num_samples (int, optional): Number of samples to use in the initial mu search. Defaults to 100.
        batch_size (int, optional): Batch size for data loading. Defaults to 1024.
        lr (float, optional): Learning rate for the optimizer. Defaults to 10.0.
        epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        torch.Tensor: The estimated friction coefficient (mu).
        tuple: A tuple containing the mu values and corresponding losses from the initial search.
    """
    # Filter the batched data
    (
        filtered_batched_initial_states,
        filtered_batched_control_inputs,
        filtered_batched_delta_times,
        filtered_batched_target_states,
    ) = filter_batched_data(
        batched_initial_states,
        batched_control_inputs,
        batched_delta_times,
        batched_target_states,
        logger=logger,
    )
    dataset = torch.utils.data.TensorDataset(
        filtered_batched_initial_states,
        filtered_batched_control_inputs,
        filtered_batched_target_states,
        filtered_batched_delta_times,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    if initial_mu_guess is None:
        initial_mu_guess, (mu_values, losses) = mu_search(
            filtered_batched_initial_states,
            filtered_batched_control_inputs,
            filtered_batched_delta_times,
            filtered_batched_target_states,
            integrator,
            range=range,
            num_samples=num_samples,
            logger=logger,
        )
    else:
        mu_values = None
        losses = None

    for param in ["mu"]:
        try:
            integrator.model_params.enable_gradients(param)
        except AttributeError:
            pass

    integrator.model_params.params["mu"] = torch.tensor(initial_mu_guess)
    optimizer = torch.optim.SGD(integrator.parameters(), lr=lr, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.9
    )

    losses = torch.zeros(epochs)
    val_losses = torch.zeros(epochs)

    for i in trange(epochs):
        batch_losses = 0
        for initial, inputs, targets, dts in dataloader:
            integrator.train()
            optimizer.zero_grad()
            output_states = integrator(initial, inputs, dts)
            loss = yaw_normalized_loss(output_states, targets)
            loss.backward()
            optimizer.step()
            batch_losses += loss.item() / initial.shape[0]
        losses[i] = batch_losses
        with torch.no_grad():
            output_states = integrator(
                batched_initial_states, batched_control_inputs, batched_delta_times
            )
            val_loss = yaw_normalized_loss(output_states, batched_target_states)
            val_losses[i] = val_loss / len(dataloader)
        scheduler.step(val_loss)

    return integrator.model_params.params["mu"], (losses, val_losses)
