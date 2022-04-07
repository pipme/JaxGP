import jax.numpy as jnp

from .config import Config
from .helpers import dataclass


@dataclass(frozen=False)
class InducingPoints:
    num_inducing: int
    D: int

    def __post_init__(self):
        self._params = {
            "inducing_points": jnp.zeros((self.num_inducing, self.D)),
        }

    @property
    def params(self) -> dict:
        return {
            "inducing_points": jnp.zeros((self.num_inducing, self.D)),
        }

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def transforms(self):
        return {"inducing_points": Config.identity_bijector}
