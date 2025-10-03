"""Compatibility layer providing JAX-like APIs with NumPy fallbacks."""

from __future__ import annotations

import numpy as _np

try:  # pragma: no cover - prefer JAX when available
    import jax
    import jax.numpy as jnp  # type: ignore
    import jax.nn as jnn  # type: ignore
    import jax.random as jr  # type: ignore
    Array = jax.Array
    USING_JAX = True
except ImportError:  # pragma: no cover - fallback path
    jnp = _np  # type: ignore
    USING_JAX = False

    class _NN:
        @staticmethod
        def softplus(x):
            return _np.log1p(_np.exp(-_np.abs(x))) + _np.maximum(x, 0.0)

    jnn = _NN()  # type: ignore

    class _Random:
        @staticmethod
        def PRNGKey(seed: int):
            return _np.random.default_rng(seed)

        @staticmethod
        def split(key):
            seed1 = int(key.integers(0, 2**32 - 1))
            seed2 = int(key.integers(0, 2**32 - 1))
            return _np.random.default_rng(seed1), _np.random.default_rng(seed2)

        @staticmethod
        def multivariate_normal(key, mean, cov, shape):
            size = int(shape[0]) if shape else None
            return key.multivariate_normal(mean, cov, size=size)

        @staticmethod
        def normal(key, shape):
            return key.normal(size=shape)

    jr = _Random()  # type: ignore
    Array = _np.ndarray  # type: ignore

__all__ = ["jnp", "jnn", "jr", "Array", "USING_JAX"]
