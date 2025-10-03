import torch

ANGLE_INDICES = [4]
all_indices = list(range(7))
non_angle_indices = [i for i in all_indices if i not in ANGLE_INDICES]


def yaw_normalized_loss(output_states, target_states):
    normal_loss = torch.nn.functional.l1_loss(
        output_states[..., non_angle_indices],
        target_states[..., non_angle_indices],
        reduction="sum",
    )
    angular_loss = torch.sum(
        torch.abs(
            torch.atan2(
                torch.sin(
                    output_states[..., ANGLE_INDICES]
                    - target_states[..., ANGLE_INDICES]
                ),
                torch.cos(
                    output_states[..., ANGLE_INDICES]
                    - target_states[..., ANGLE_INDICES]
                ),
            )
        )
    )
    return normal_loss + angular_loss


def yaw_normalized_loss_per_item(output_states, target_states):
    """Calculate the yaw normalized loss per item in the batch"""
    normal_loss_p1 = torch.nn.functional.l1_loss(
        output_states[..., non_angle_indices],
        target_states[..., non_angle_indices],
        reduction="none",
    ).sum([1, 2])
    angular_loss = torch.abs(
        torch.atan2(
            torch.sin(
                output_states[..., ANGLE_INDICES] - target_states[..., ANGLE_INDICES]
            ),
            torch.cos(
                output_states[..., ANGLE_INDICES] - target_states[..., ANGLE_INDICES]
            ),
        )
    ).sum([1])
    return normal_loss_p1 + angular_loss


def yaw_normalized_loss_per_element(output_states, target_states):
    """Assuming the data is (B, T, D), return the loss per dimension in the state, (B,D)"""
    losses = torch.zeros(
        output_states.shape[0], output_states.shape[2], device=output_states.device
    )
    normal_loss = torch.nn.functional.l1_loss(
        output_states[..., non_angle_indices],
        target_states[..., non_angle_indices],
        reduction="none",
    ).sum(1)
    angular_loss = torch.abs(
        torch.atan2(
            torch.sin(
                output_states[..., ANGLE_INDICES] - target_states[..., ANGLE_INDICES]
            ),
            torch.cos(
                output_states[..., ANGLE_INDICES] - target_states[..., ANGLE_INDICES]
            ),
        )
    ).sum(1)
    losses[..., non_angle_indices] = normal_loss
    losses[..., ANGLE_INDICES] = angular_loss
    return losses
