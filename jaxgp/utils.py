import copy
from typing import Any, Dict, TypeVar, Union

import jax
import numpy as np

from .abstractions import InducingPoints
from .helpers import Array

KeyType = TypeVar("KeyType")


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


def copy_dict_structure(params: Dict) -> Dict:
    # Copy dictionary structure
    container = copy.deepcopy(params)
    # Set all values to None
    container = jax.tree_map(lambda _: None, container)
    return container


def deep_update(
    mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]
) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def deep_update_no_new_key(
    mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]
) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update_no_new_key(
                    updated_mapping[k], v
                )
            else:
                if k in updated_mapping:
                    updated_mapping[k] = v
    return updated_mapping


def save_params_npy(params: Dict, file_path: str) -> None:
    params = jax.tree_util.tree_map(lambda v: np.array(v), params)
    np.save(file_path, params)
