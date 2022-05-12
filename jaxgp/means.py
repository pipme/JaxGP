from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import jax.numpy as jnp

from .config import Config
from .helpers import Array, dataclass, field


@dataclass
class MeanFunction(metaclass=ABCMeta):
    output_dim: Optional[int] = 1
    name: str = "Mean function"

    @abstractmethod
    def __call__(self, x: Array, params: Dict) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}\n\t Output dimension: {self.output_dim}"

    @property
    @abstractmethod
    def params(self) -> Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def transforms(self) -> Dict:
        raise NotImplementedError


@dataclass
class Zero(MeanFunction):
    output_dim: Optional[int] = 1
    name: str = "Zero mean function"

    def __call__(self, x: Array, params: Optional[Dict] = None) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)

    @property
    def params(self) -> Dict:
        return {}

    @property
    def transforms(self) -> Dict:
        return {}


@dataclass
class Constant(MeanFunction):
    output_dim: int = 1
    name: str = "Constant mean function"
    _params: Dict = field(
        default_factory=lambda: {"mean_const": jnp.array([1.0])}
    )

    def __call__(self, x: Array, params: Dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * params["mean_const"]

    @property
    def params(self) -> Dict:
        return {"mean_const": jnp.array([1.0])}

    @property
    def transforms(self) -> Dict:
        return {"mean_const": Config.identity_bijector}


@dataclass
class Quadratic(MeanFunction):
    r"""Quadratic mean function

    .. math::
        m(x) = m_0 - \frac{1}{2} \sum_{i = 1}^{D} (\frac{x^{(i)} - x_{m}^{(i)}}{\omega^{(i)}})^2

    mean_const: The parameter :math:`m_0`.
    scale: The parameter :math:`\omega`.
    """
    input_dim: int = 1
    name: str = "Quadratic mean function"

    def __call__(self, x: Array, params: Dict) -> Array:
        return params["mean_const"] - 0.5 * jnp.sum(
            ((x - params["xm"]) / params["scale"]) ** 2, -1, keepdims=True
        )

    @property
    def params(self) -> Dict:
        return {
            "mean_const": jnp.array([0.0]),
            "scale": jnp.array([1.0] * self.input_dim),
            "xm": jnp.array([0.0] * self.input_dim),
        }

    @property
    def transforms(self) -> Dict:
        return {
            "mean_const": Config.identity_bijector,
            "scale": Config.strict_positive_bijector,
            "xm": Config.identity_bijector,
        }
