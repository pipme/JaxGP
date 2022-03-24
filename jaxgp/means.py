import abc
from typing import Dict, Optional

import jax.numpy as jnp

from .config import Config
from .helpers import Array, dataclass, field


@dataclass
class MeanFunction:
    output_dim: Optional[int] = 1
    name: str = "Mean function"
    _params: Dict = field(default_factory=dict)

    @abc.abstractmethod
    def __call__(self, x: Array, params: Dict) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
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
    name: str = "Zero mean function"
    _params: Dict = field(default_factory=dict)

    def __call__(self, x: Array, params: Optional[Dict] = None) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)

    @property
    def transforms(self) -> Dict:
        return {}


@dataclass
class Constant(MeanFunction):
    output_dim: int = 1
    name: str = "Constant mean function"
    _params: Dict = field(default_factory=lambda: {"constant": jnp.array(1.0)})

    def __call__(self, x: Array, params: Dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * params["constant"]

    @property
    def transforms(self) -> Dict:
        return {"constant": Config.identity_bijector}


@dataclass
class Quadratic(MeanFunction):
    r"""Quadratic mean function

    .. math::
        m(x) = m_0 - \frac{1}{2} \sum_{i = 1}^{D} (\frac{x^{(i)} - x_{m}^{(i)}}{\omega^{(i)}})^2

    scale: The parameter :math:`\omega`.
    """
    input_dim: int = 1
    name: str = "Quadratic mean function"

    def __post_init__(self) -> None:
        self._params = {
            "m0": jnp.array([0.0]),
            "scale": jnp.array([1.0] * self.input_dim),
            "xm": jnp.array([0.0] * self.input_dim),
        }

    def __call__(self, x: Array, params: Dict) -> Array:
        return params["m0"] - 0.5 * jnp.sum(
            ((x - params["xm"]) / params["scale"]) ** 2, -1, keepdims=True
        )

    @property
    def transforms(self) -> Dict:
        return {
            "m0": Config.identity_bijector,
            "scale": Config.positive_bijector,
            "xm": Config.identity_bijector,
        }
