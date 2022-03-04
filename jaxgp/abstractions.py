from typing import Optional

import jax.numpy as jnp
from chex import dataclass

from .config import Config


@dataclass
class InducingPoints:
    num_inducing: int
    D: int

    def __post_init__(self):
        self._params = {
            "inducing_points": jnp.zeros((self.num_inducing, self.D)),
        }

    @property
    def params(self) -> dict:
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def transforms(self):
        return {"inducing_points": Config.identity_bijector}
