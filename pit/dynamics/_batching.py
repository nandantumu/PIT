from __future__ import annotations

from typing import Callable, Tuple, TypeVar

import torch

TensorLike = TypeVar("TensorLike", bound=torch.Tensor)


def ensure_batch(tensor: TensorLike) -> Tuple[TensorLike, Callable[[TensorLike], TensorLike]]:
    """Ensure a tensor has a batch dimension.

    The helper promotes one-dimensional tensors to batched tensors by
    unsqueezing a leading dimension.  A callable is returned that can be
    applied to tensors with the resulting shape to restore the original
    dimensionality.

    Args:
        tensor: A tensor with shape ``(dim,)`` or ``(batch, dim)``.

    Returns:
        A tuple containing the (potentially) batched tensor and a callable to
        restore tensors with matching shape back to their original
        dimensionality.
    """

    if tensor.ndim == 1:
        batched = tensor.unsqueeze(0)

        def restore(result: TensorLike) -> TensorLike:
            return result.squeeze(0)

        return batched, restore

    return tensor, lambda result: result


__all__ = ["ensure_batch"]
