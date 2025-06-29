import torch


def yaw_normalized_loss(output_states, target_states):
    normal_loss_p1 = torch.nn.functional.l1_loss(
        output_states[..., :3], target_states[..., :3], reduction="sum"
    )
    normal_loss_p2 = torch.nn.functional.l1_loss(
        output_states[..., 4:], target_states[..., 4:], reduction="sum"
    )
    angular_loss = torch.sum(
        torch.abs(
            torch.atan2(
                torch.sin(output_states[..., 3] - target_states[..., 3]),
                torch.cos(output_states[..., 3] - target_states[..., 3]),
            )
        )
    )
    return normal_loss_p1 + normal_loss_p2 + angular_loss


def yaw_normalized_loss_per_item(output_states, target_states):
    """Calculate the yaw normalized loss per item in the batch"""
    normal_loss_p1 = torch.nn.functional.l1_loss(
        output_states[..., :3], target_states[..., :3], reduction="none"
    ).sum([1, 2])
    normal_loss_p2 = torch.nn.functional.l1_loss(
        output_states[..., 4:], target_states[..., 4:], reduction="none"
    ).sum([1, 2])
    angular_loss = torch.abs(
        torch.atan2(
            torch.sin(output_states[..., 3] - target_states[..., 3]),
            torch.cos(output_states[..., 3] - target_states[..., 3]),
        )
    ).sum([1])
    return normal_loss_p1 + normal_loss_p2 + angular_loss
