import collections.abc
import copy
import time
from typing import Any, Dict, NamedTuple, Optional, Union

import jax.numpy as jnp
import jaxopt

import jaxgp as jgp
from jaxgp import GPR, SGPR, HeteroskedasticSGPR
from jaxgp.utils import deep_update, deep_update_no_new_key


def train_model(
    model: Union[GPR, SGPR, HeteroskedasticSGPR],
    init_params: Optional[Dict] = None,
    fixed_params: Optional[Dict] = None,
    tol: Optional[float] = None,
    options: Optional[float] = None,
    transforms_jitted: Optional[tuple[Any, Any]] = None,
    return_soln: Optional[bool] = False,
    **kwargs,
) -> NamedTuple:
    if transforms_jitted is not None:
        params = model.params
        constrain_trans, unconstrain_trans = transforms_jitted
    else:
        params, constrain_trans, unconstrain_trans = jgp.initialise(model)
    raw_params = unconstrain_trans(params)
    if isinstance(model, GPR):
        neg_elbo = model.build_mll(sign=-1.0)
    else:
        neg_elbo = model.build_elbo(sign=-1.0)

    if init_params is not None:
        params = deep_update(params, init_params)
        raw_params = unconstrain_trans(params)

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

    ts = time.time()
    print("Initial negative elbo = ", obj_fun(raw_params))
    solver = jaxopt.ScipyMinimize(
        fun=obj_fun, jit=True, tol=tol, options=options
    )
    if "bounds" not in kwargs:
        soln = solver._run(raw_params, bounds=None, **kwargs)
    else:
        soln = solver._run(raw_params, **kwargs)
    print(f"After optimization negative elbo = {soln.state.fun_val}")
    t2 = time.time() - ts
    print("Time of jaxopt: ", t2)
    if return_soln:
        return soln
    else:
        final_params = constrain_trans(soln.params)
        return final_params


def train_model_separate(
    model: Union[GPR, SGPR, HeteroskedasticSGPR],
    params: Dict,
    diff_params: Dict,
    fixed_params: Optional[Dict] = None,
    tol: Optional[float] = None,
    options: Optional[float] = None,
    transforms_jitted: Optional[tuple[Any, Any]] = None,
    return_soln: Optional[bool] = False,
    **kwargs,
):
    if transforms_jitted is not None:
        constrain_trans, unconstrain_trans = transforms_jitted
    else:
        _, constrain_trans, unconstrain_trans = jgp.initialise(model)

    if isinstance(model, GPR):
        neg_elbo = model.build_mll(sign=-1.0)
    else:
        neg_elbo = model.build_elbo(sign=-1.0)

    if fixed_params is None:
        # Only work for optimizing inducing points
        mask = diff_params["inducing_points"]
        diff_raw_params = unconstrain_trans(
            {"inducing_points": params["inducing_points"][mask]}
        )
        fixed_raw_params = unconstrain_trans(params)
        fixed_raw_params["inducing_points"] = fixed_raw_params[
            "inducing_points"
        ][~mask]
    else:
        diff_raw_params = unconstrain_trans(diff_params)
        fixed_raw_params = unconstrain_trans(fixed_params)

    def obj_fun(diff_raw_params):
        raw_params = combine_dict(diff_raw_params, fixed_raw_params)
        return neg_elbo(raw_params)

    ts = time.time()
    print("Initial negative elbo = ", obj_fun(diff_raw_params))
    solver = jaxopt.ScipyMinimize(
        fun=obj_fun, jit=True, tol=tol, options=options
    )
    if "bounds" not in kwargs:
        soln = solver._run(diff_raw_params, bounds=None, **kwargs)
    else:
        soln = solver._run(diff_raw_params, **kwargs)
    print(f"After optimization negative elbo = {soln.state.fun_val}")
    t2 = time.time() - ts
    print("Time of jaxopt: ", t2)
    if return_soln:
        return soln
    else:
        diff_raw_params = soln.params
        raw_params = combine_dict(diff_raw_params, fixed_raw_params)
        final_params = constrain_trans(raw_params)
        return final_params


def combine_dict(*args):
    result = {}
    for d in args:
        for k in d.keys():
            if k not in result.keys():
                result[k] = d[k]
            else:
                if type(d[k]) is dict:
                    result[k] = combine_dict(result[k], d[k])
                else:
                    result[k] = jnp.concatenate([result[k], d[k]])
    return result
