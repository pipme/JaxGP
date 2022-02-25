from copy import deepcopy
import jax
from typing import Tuple, Dict


def initialise(obj) -> Tuple[Dict, Dict, Dict]:
    params = obj.params
    # constrainers, unconstrainers = build_transforms(params)
    # return params, constrainers, unconstrainers
    return params, None, None


def copy_dict_structure(params: dict) -> dict:
    # Copy dictionary structure
    prior_container = deepcopy(params)
    # Set all values to zero
    prior_container = jax.tree_map(lambda _: None, prior_container)
    return prior_container
