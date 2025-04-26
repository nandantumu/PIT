import torch
from ..loss import yaw_normalized_loss, yaw_normalized_loss_per_item
from ..data.filtering import filter_batched_data
from tqdm.auto import trange
import numpy as np

try:
    import cma
except ImportError as e:
    raise ImportError(
        "Please install the 'cma' package (pip install cma) to use CMA-ES functionality."
    ) from e


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
    mu_values = torch.linspace(
        range[0], range[1], num_samples, device=batched_delta_times.device
    )
    losses = torch.zeros_like(mu_values)
    for i, mu in enumerate(mu_values):
        integrator.model_params.params["mu"] = torch.tensor(
            mu, device=batched_delta_times.device
        )
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
    try:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    except ValueError:
        return float(initial_mu_guess), (None, None)
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
        losses = None
        initial_mu_guess = torch.tensor(
            initial_mu_guess, dtype=torch.float64, device=batched_delta_times.device
        )

    for param in ["mu"]:
        try:
            integrator.model_params.enable_gradients(param)
        except AttributeError:
            pass

    # integrator.model_params.params["mu"] = torch.tensor(
    #     initial_mu_guess, device=batched_delta_times.device
    # )
    optimizer = torch.optim.SGD(integrator.parameters(), lr=lr, momentum=0.8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.9
    )

    losses = torch.zeros(epochs)
    val_losses = torch.zeros(epochs)

    for i in trange(epochs):
        batch_losses = 0
        orig_count = 0
        removed = 0
        for initial, inputs, targets, dts in dataloader:
            integrator.train()
            optimizer.zero_grad()
            output_states = integrator(initial, inputs, dts)
            loss = yaw_normalized_loss(output_states, targets)
            # Eliminate the Nan values from the loss, only calculate on the valid values
            nan_mask = torch.isnan(loss)
            loss = loss[~nan_mask]
            # compute how many items were dropped by the NaN filter and log the fraction
            orig_count += initial.shape[0]
            removed += initial.shape[0] - loss.numel()
            loss.backward()
            optimizer.step()
            batch_losses += loss.item() / initial.shape[0]
        frac_removed = removed / orig_count
        # find the first position where loss is NaN

        logger.info(f"Example NaN item: {output_states}")

        if logger is not None:
            logger.info(
                f"Fraction of loss items removed due to NaNs: {frac_removed:.3f}"
            )
        losses[i] = batch_losses
        with torch.no_grad():
            output_states = integrator(
                batched_initial_states, batched_control_inputs, batched_delta_times
            )
            val_loss = yaw_normalized_loss(output_states, batched_target_states)
            val_losses[i] = val_loss / len(dataloader)
        scheduler.step(val_loss)

    return integrator.model_params.params["mu"], (losses, val_losses)


def gradient_search(
    batched_initial_states,
    batched_control_inputs,
    batched_delta_times,
    batched_target_states,
    integrator,
    batch_size=1024,
    lr=0.1,
    epochs: int = 100,
    logger=None,
    param_ranges: dict = None,
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

    # Check if the filtered data is empty
    if filtered_batched_initial_states.shape[0] == 0:
        logger.warning("Filtered data is empty. Returning initial parameters.")
        return integrator.model_params, (None, None)

    dataset = torch.utils.data.TensorDataset(
        filtered_batched_initial_states,
        filtered_batched_control_inputs,
        filtered_batched_target_states,
        filtered_batched_delta_times,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.SGD(integrator.parameters(), lr=lr, momentum=0.8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.9
    )

    losses = torch.zeros(epochs)
    val_losses = torch.zeros(epochs)
    orig_count = 0
    removed = 0

    for i in trange(epochs):
        batch_losses = 0

        for initial, inputs, targets, dts in dataloader:
            integrator.train()
            optimizer.zero_grad()
            output_states = integrator(initial, inputs, dts)
            loss = yaw_normalized_loss_per_item(output_states, targets)
            # Eliminate the Nan values from the loss, only calculate on the valid values
            nan_mask = torch.isnan(loss)
            loss = loss[~nan_mask]
            # compute how many items were dropped by the NaN filter and log the fraction
            orig_count += initial.shape[0]
            removed += initial.shape[0] - loss.numel()
            if loss.numel() == 0:
                continue  # Skip this batch if all losses are NaN
            loss = loss.mean()
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
    frac_removed = removed / orig_count
    if logger is not None:
        logger.info(
            f"Fraction of loss items removed due to NaNs: {removed}/{orig_count} = {frac_removed:.3f}"
        )
    return integrator.model_params, (losses, val_losses)


def cma_search(
    batched_initial_states,
    batched_control_inputs,
    batched_delta_times,
    batched_target_states,
    integrator,
    param_ranges: dict,
    sigma=None,
    popsize=None,
    maxiter=100,
    logger=None,
):
    """
    Perform CMA-ES search to estimate the friction coefficient (mu) for a given integrator model.

    Args:
        batched_initial_states (torch.Tensor): Batched initial states of the system.
        batched_control_inputs (torch.Tensor): Batched control inputs applied to the system.
        batched_delta_times (torch.Tensor): Batched time intervals between states.
        batched_target_states (torch.Tensor): Batched target states to be achieved.
        integrator (Integrator): The integrator model used for state prediction.
        param_ranges (dict, optional): Search bounds for parameters. Defaults to None (default bounds for all params).
        sigma (float, optional): Initial standard deviation for CMA-ES. Defaults to half the range.
        popsize (int, optional): Population size for CMA-ES. Defaults to None (library default).
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        list: The estimated parameters.
        tuple: Placeholder tuple (None, None) for consistency with other methods.
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

    # Prepare CMA-ES over all parameters in the ParameterGroup
    param_names = list(integrator.model_params.params.keys())
    num_params = len(param_names)
    # define bounds for each parameter using param_ranges dict
    if param_ranges is None:
        # default bounds for all params
        param_ranges = {name: (0.1, 1.0) for name in param_names}
    lower, upper = zip(*[param_ranges[name] for name in param_names])

    # initial guess is midpoint for each parameter
    x0 = [(low + high) / 2.0 for low, high in zip(lower, upper)]
    # initial sigma
    if sigma is None:
        sigma0 = np.mean([(high - low) / 2.0 for low, high in zip(lower, upper)])
    else:
        sigma0 = sigma if hasattr(sigma, "__iter__") else [sigma] * num_params
    # setup CMA-ES
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {"bounds": [lower, upper], "popsize": popsize, "maxiter": maxiter}
    )

    # objective function
    def objective(x):
        # assign CMA values to model parameters
        for val, name in zip(x, param_names):
            integrator.model_params.params[name].data = torch.tensor(
                val, device=batched_delta_times.device
            )
        with torch.no_grad():
            outs = integrator(
                filtered_batched_initial_states,
                filtered_batched_control_inputs,
                filtered_batched_delta_times,
            )
            losses = yaw_normalized_loss_per_item(outs, filtered_batched_target_states)
            # Replace NaN values with a large number
            losses = torch.where(
                torch.isnan(losses), torch.tensor(1e10, device=losses.device), losses
            )
            return float(losses.mean())

    # run optimization
    es.optimize(objective)
    best_params = es.result.xbest
    # update model parameters with best result
    for val, name in zip(best_params, param_names):
        if name in integrator.model_params.params:
            integrator.model_params.params[name].data = torch.tensor(
                val, device=batched_delta_times.device
            )
    if logger:
        logger.info(f"CMA-ES converged params: {dict(zip(param_names, best_params))}")
    return integrator.model_params, (None, None)
