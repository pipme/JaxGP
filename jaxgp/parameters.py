from copy import deepcopy
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from .config import Config


def initialise(obj: object) -> Tuple[Dict, Callable, Callable]:
    params = obj.params
    constrain_trans, unconstrain_trans = build_transforms(obj.transforms)
    return params, constrain_trans, unconstrain_trans


def copy_dict_structure(params: Dict) -> Dict:
    # Copy dictionary structure
    prior_container = deepcopy(params)
    # Set all values to zero
    prior_container = jax.tree_map(lambda _: None, prior_container)
    return prior_container


def sort_dict(base_dict: Dict) -> Dict:
    return dict(sorted(base_dict.items()))


def build_transforms(transforms: Dict) -> Tuple[Callable, Callable]:
    transforms = sort_dict(transforms)

    def constrain_trans(params: Dict) -> Dict:
        params = sort_dict(params)
        transforms_ = {}
        for k in params.keys():
            transforms_[k] = transforms[k]

        def transform_param(param, transform):  # type: ignore
            if isinstance(transform, tfp.bijectors.Bijector):
                return jnp.array(transform.forward(param))
            else:
                return param

        return jax.tree_util.tree_multimap(
            transform_param, params, transforms_
        )

    def unconstrain_trans(params: Dict) -> Dict:
        params = sort_dict(params)
        transforms_ = {}
        for k in params.keys():
            transforms_[k] = transforms[k]

        def transform_param(param, transform):  # type: ignore
            if isinstance(transform, tfp.bijectors.Bijector):
                return jnp.array(transform.inverse(param))
            else:
                return param

        return jax.tree_util.tree_multimap(
            transform_param, params, transforms_
        )

    return constrain_trans, unconstrain_trans
