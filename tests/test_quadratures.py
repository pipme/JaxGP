# -*- coding: utf-8 -*-
# mypy: ignore-errors

import numpy as np

from jaxgp.quadratures import gauss_hermite_quadrature


def test_gauss_hermite_quadrature():
    def f(*args, **kwargs):
        return 1

    v = gauss_hermite_quadrature(f, 1.1, 2.3)
    np.testing.assert_allclose(v, 1.0)
