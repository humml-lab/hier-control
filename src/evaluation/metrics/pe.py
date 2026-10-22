from typing import Dict, Optional

import torch as tc

from src.models.mtf_model import MTFModel


def n_step_ahead_latent_predictions(
    model: MTFModel,
    data: tc.Tensor,
    n_steps_ahead: int,
    cumulative_timeseries_index: int,
    measurement_index: int,
    external_inputs: Optional[tc.Tensor] = None,
    min_context_length: int = 20,
    max_batch_size: int = -1,
) -> Dict[str, tc.Tensor]:
    """
    Computes n-step ahead predictions in latent space only, without decoding to observation space.

    For each valid position in the time series, uses exactly min_context_length points as context
    to encode the initial latent state, then predicts n_steps_ahead into the future autoregressively.
    Returns predictions and ground truth in latent space only.

    Args:
        model (MTFModel): The trained MTF model used for encoding and prediction.
        data (tc.Tensor): Time series data of shape (T, N), where T is the number of time steps and N is the number of features.
        n_steps_ahead (int): Number of steps ahead to predict autoregressively.
        cumulative_timeseries_index (int): Index identifying which timeseries this is (for model parameter selection).
        measurement_index (int): Index identifying which measurement type this is (for encoder selection).
        external_inputs (Optional[tc.Tensor]): Optional external inputs of shape (T, M), where M is the number of external features. Default is None.
        min_context_length (int): Minimum number of time steps to use as context for encoding. Default is 20.
        max_batch_size (int): Maximum batch size for processing prediction windows. Use -1 to process all windows in a single batch. Default is 64.

    Returns:
        Dict[str, tc.Tensor]: Dictionary containing:
            - 'predicted_latents': Predicted latent states (tensor of shape (n_windows, latent_dim))
            - 'ground_truth_latents': Ground truth latent states (tensor of shape (n_windows, latent_dim))
            - 'prediction_positions': Time positions of predictions (tensor of shape (n_windows,))
    """
    was_training = model.training
    if was_training:
        model.eval()

    T, N = data.shape

    # Calculate valid starting positions for prediction windows
    min_start_pos = min_context_length
    max_start_pos = T - n_steps_ahead

    if max_start_pos <= min_start_pos:
        raise ValueError(
            f"Time series too short. Need at least {min_start_pos + n_steps_ahead} time steps, got {T}."
        )

    # Create all prediction windows
    valid_positions = list(range(min_start_pos, max_start_pos + 1))
    n_windows = len(valid_positions)

    # Handle max_batch_size = -1 (single batch)
    if max_batch_size == -1:
        max_batch_size = n_windows

    # Construct DSR parameters once for all batches
    dsr_params = model.dsr_model.construct_params(
        cumulative_timeseries_index=cumulative_timeseries_index
    )

    # ====================================================================
    # Process predictions in batches with proper overlapping context
    # ====================================================================

    all_latent_predictions = []
    all_latent_ground_truth = []

    for batch_start in range(0, n_windows, max_batch_size):
        batch_end = min(batch_start + max_batch_size, n_windows)
        batch_positions = valid_positions[batch_start:batch_end]

        # ================================================================
        # STEP 1: Prepare batched context windows and targets
        # ================================================================

        context_windows = []
        target_context_windows = []
        batch_ext_inputs = []

        for pos in batch_positions:
            # Context window: exactly min_context_length points leading up to pos
            context_start = pos - min_context_length
            context_window = data[context_start:pos]  # (min_context_length, N)
            context_windows.append(context_window)

            # Target position
            target_pos = pos + n_steps_ahead - 1

            # Target context window: exactly min_context_length points leading up to target_pos + 1
            # This gives us the proper encoding of the target state
            target_context_start = target_pos + 1 - min_context_length
            target_context_window = data[
                target_context_start : target_pos + 1
            ]  # (min_context_length, N)
            target_context_windows.append(target_context_window)

            # External inputs for the prediction steps
            if external_inputs is not None:
                ext_inp = external_inputs[pos : pos + n_steps_ahead]  # (n_steps_ahead, M)
                batch_ext_inputs.append(ext_inp)

        # Stack context windows into batch tensors and move to device
        batch_context = tc.stack(context_windows, dim=1).to(
            model.device
        )  # (min_context_length, batch_size,  N)
        batch_target_context = tc.stack(target_context_windows, dim=1).to(
            model.device
        )  # (min_context_length, batch_size, N)

        if external_inputs is not None:
            batch_ext_inputs = tc.stack(batch_ext_inputs, dim=1).to(
                model.device
            )  # (n_steps_ahead, batch_size, M)
        else:
            batch_ext_inputs = None

        # ================================================================
        # STEP 2: Encode context windows to get initial and target latent states
        # ================================================================

        # Encode initial contexts: (batch_size, min_context_length, N) -> (batch_size, N, min_context_length)
        _, _, encoded_contexts_mean = model.encoder_model(
            batch_context, measurement_index, calculate_entropy=False
        )

        # Extract initial states (last encoded state from each context)
        batch_initial_states = encoded_contexts_mean[-1]  # (batch_size, latent_dim)

        # Encode target contexts to get target latent states
        _, _, encoded_target_contexts_mean = model.encoder_model(
            batch_target_context, measurement_index, calculate_entropy=False
        )

        # Extract target latent states (last encoded state from each target context)
        batch_target_latents = encoded_target_contexts_mean[-1]  # (batch_size, latent_dim)

        # ================================================================
        # STEP 3: Perform autoregressive latent prediction (batched)
        # ================================================================

        current_states = batch_initial_states  # (batch_size, latent_dim)

        # Predict n_steps_ahead autoregressively
        for step in range(n_steps_ahead):
            # Get external inputs for this step
            step_ext_input = None
            if batch_ext_inputs is not None:
                step_ext_input = batch_ext_inputs[step]

            # Vectorized DSR step: process entire batch at once
            next_states = model.dsr_model.latent_step.forward(
                current_states,
                step_ext_input,
                dsr_params,
                model.dsr_model.hyperparameters,
            )

            current_states = next_states

        batch_predicted_latents = current_states  # (batch_size, latent_dim)

        # ================================================================
        # STEP 4: Store predictions and ground truth
        # ================================================================

        all_latent_predictions.append(batch_predicted_latents)
        all_latent_ground_truth.append(batch_target_latents)

    # ====================================================================
    # STEP 5: Combine results from all batches
    # ====================================================================

    # Combine all predictions
    all_latent_predictions = tc.cat(all_latent_predictions, dim=0)  # (n_windows, latent_dim)
    all_latent_ground_truth = tc.cat(all_latent_ground_truth, dim=0)  # (n_windows, latent_dim)

    # Create time positions for predictions
    prediction_positions = tc.tensor(
        [pos + n_steps_ahead - 1 for pos in valid_positions], device=model.device
    )

    # Restore training mode if it was originally training
    if was_training:
        model.train()

    return all_latent_predictions, all_latent_ground_truth, prediction_positions
