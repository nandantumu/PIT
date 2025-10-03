from __future__ import annotations

from typing import Callable, Tuple

from .._compat import Array, jnp


def ensure_batch(tensor: Array) -> Tuple[Array, Callable[[Array], Array]]:
    """Ensure a tensor has a batch dimension.

    Promotes one-dimensional arrays to batched arrays by adding a leading
    dimension.  A callable is returned that can be applied to arrays with the
    resulting shape to restore the original dimensionality.
    """

    if tensor.ndim == 1:
        batched = jnp.expand_dims(tensor, axis=0)

        def restore(result: Array) -> Array:
            return jnp.squeeze(result, axis=0)

        return batched, restore

    return tensor, lambda result: result


__all__ = ["ensure_batch"]
