import collections.abc
import copy
from typing import Any, Dict, NamedTuple, Optional, TypeVar, Union

import jaxopt
from pydantic.utils import deep_update

import jaxgp as jgp
from jaxgp import SGPR, HeteroskedasticSGPR

KeyType = TypeVar("KeyType")


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


def train_model(
    model: Union[SGPR, HeteroskedasticSGPR],
    fixed_params: Optional[Dict] = None,
) -> NamedTuple:
    params, constrain_trans, unconstrain_trans = jgp.initialise(model)
    raw_params = unconstrain_trans(params)
    neg_elbo = model.build_elbo(sign=-1.0)

    if fixed_params is not None:
        # hack to update, better ways?
        params = deep_update(params, fixed_params)
        raw_params = unconstrain_trans(params)
        fixed_raw_params = copy.deepcopy(fixed_params)
        fixed_raw_params = deep_update_no_new_key(fixed_raw_params, raw_params)

        def obj_fun(raw_params):  # type: ignore
            raw_params = deep_update(raw_params, fixed_raw_params)
            return neg_elbo(raw_params)

    else:
        obj_fun = neg_elbo

    print("Initial negative elbo = ", obj_fun(raw_params))
    solver = jaxopt.ScipyMinimize(
        fun=obj_fun, jit=True, options={"disp": True}
    )
    soln = solver.run(raw_params)
    return soln
