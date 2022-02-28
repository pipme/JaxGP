import abc
from signal import raise_signal
from typing import Dict, Optional

import jax.numpy as jnp
from chex import dataclass

from .types import Array
from .config import Config


@dataclass(repr=False)
class MeanFunction:
    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean function"

    @abc.abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}\n\t Output dimension: {self.output_dim}"

    @property
    @abc.abstractmethod
    def params(self) -> Dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def transforms(self) -> Dict:
        raise NotImplementedError


@dataclass(repr=False)
class Zero(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean function"

    def __call__(self, x: Array, params: dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)

    @property
    def params(self) -> dict:
        return {}

    @property
    def transforms(self) -> Dict:
        return {}


@dataclass(repr=False)
class Constant(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Constant mean function"
    _params: Optional[Dict] = None

    def __post_init__(self):
        self._params = {"constant": jnp.array(1.0)}

    def __call__(self, x: Array, params: Dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * params["constant"]

    @property
    def params(self) -> dict:
        return self._params

    @property
    def transforms(self) -> Dict:
        return {"constant": Config.identity_bijector}
