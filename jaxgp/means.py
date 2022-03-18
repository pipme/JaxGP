import abc
from typing import Dict, Optional

import jax.numpy as jnp

from .config import Config
from .helpers import Array, dataclass, field


@dataclass
class MeanFunction:
    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean function"
    _params: Optional[Dict] = field(default_factory=dict)

    @abc.abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}\n\t Output dimension: {self.output_dim}"

    @property
    def params(self) -> Dict:
        return self._params

    @property
    @abc.abstractmethod
    def transforms(self) -> Dict:
        raise NotImplementedError


@dataclass
class Zero(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean function"
    _params: Optional[dict] = field(default_factory=dict)

    def __call__(self, x: Array, params: Dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)

    @property
    def transforms(self) -> Dict:
        return {}


@dataclass
class Constant(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Constant mean function"
    _params: Optional[dict] = field(
        default_factory=lambda: {"constant": jnp.array(1.0)}
    )

    def __call__(self, x: Array, params: Dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * params["constant"]

    @property
    def transforms(self) -> Dict:
        return {"constant": Config.identity_bijector}
