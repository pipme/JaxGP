from typing import List, Dict, Union
from .abstractions import InducingPoints
from .types import Array


def concat_dictionaries(*args: List[Dict]) -> Dict:
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
        N, D = inducing_points.shape
        inducing_points_obj = InducingPoints(num_inducing=N, D=D)
        inducing_points_obj.params["inducing_points"] = inducing_points
        inducing_points = inducing_points_obj
    return inducing_points
