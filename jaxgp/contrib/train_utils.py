import collections.abc
import copy
from typing import Any, Dict, NamedTuple, Optional, Union

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
    import time

    ts = time.time()
    print("Initial negative elbo = ", obj_fun(raw_params))
    solver = jaxopt.ScipyMinimize(
        fun=obj_fun, jit=True, tol=tol, options=options
    )
    if "bounds" not in kwargs:
        soln = solver._run(raw_params, bounds=None, **kwargs)
    else:
        soln = solver._run(raw_params, **kwargs)
    t2 = time.time() - ts
    print("Time of jaxopt: ", t2)
    return soln
