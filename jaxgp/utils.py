import copy
from typing import Dict, Union

import jax

from .abstractions import InducingPoints
from .helpers import Array


def concat_dictionaries(*args: Dict) -> Dict:
    """
    Append one dictionary below another. If duplicate keys exist, then the key-value pair of the last supplied
    dictionary will be used.
    """
    result = {}
    for d in args:
        result.update(d)
    return result


def inducingpoint_wrapper(
    inducing_points: Union[InducingPoints, Array]
) -> InducingPoints:
    """
    This wrapper allows transparently passing either an InducingPoints
    object or an array specifying InducingPoints positions.
    """
    if not isinstance(inducing_points, InducingPoints):
        if inducing_points.ndim == 1:
            inducing_points = inducing_points[..., None]
        N, D = inducing_points.shape
        inducing_points_obj = InducingPoints(num_inducing=N, D=D)
        inducing_points_obj.params["inducing_points"] = inducing_points
        inducing_points = inducing_points_obj
    return inducing_points


def pytree_shape_info(params: Dict) -> None:
    # print params' shape info
    params_container = copy.deepcopy(params)
    params_container = jax.tree_map(lambda v: v.shape, params_container)
    print(params_container)
