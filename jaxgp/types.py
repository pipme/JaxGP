# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Array"]

from typing import Union

import jax.numpy as jnp
import numpy as np
from chex import dataclass

Array = Union[np.ndarray, jnp.ndarray]


@dataclass(repr=False)
class Dataset:
    X: Array
    y: Array = None

    def __post_init__(self):
        if self.y.ndim == 1:
            self.y = self.y[..., None]
        assert self.X.ndim == 2
        assert self.y.ndim == 2

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension:"
            f" {self.X.shape[1]}"
        )

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        return self.y.shape[1]
