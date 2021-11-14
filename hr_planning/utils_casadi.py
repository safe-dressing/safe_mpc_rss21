# -*- coding: utf-8 -*-
"""
Adapted from the following file, with some additional functions:

# https://github.com/befelix/safe-exploration/blob/master/safe_exploration/utils_casadi.py
Utility functions implemented with the casadi library
A set of utility functions required by various different modules / classes.
This module contains a subset of the functions implemented in the corresponding
utils.py file with the same functionality but is implemented in a way that
admits being use by a Casadi ( symbolic ) framework.

@author: tkoller
"""

import numpy as np
from casadi import mtimes, fmax, norm_2, sqrt, exp, SX, cos, sin, det, inv, vertcat, \
    horzcat, trace, MX, Function
from casadi import reshape as cas_reshape
from IPython import embed


def compute_remainder_overapproximations(q, l_mu, l_sigma):
    """ Compute symbolically the (hyper-)rectangle over-approximating
        the lagrangians of mu and sigma.

    Parameters
    ----------
    q: n_s x n_s ndarray[casadi.SX.sym]
        The shape matrix of the current state ellipsoid
    l_mu: n x 0 numpy 1darray[float]
        The lipschitz constants for the gradients of the predictive mean
    l_sigma n x 0 numpy 1darray[float]
        The lipschitz constans on the predictive variance

    Returns
    -------
    u_mu: n_s x 0 numpy 1darray[casadi.SX.sym]
        The upper bound of the over-approximation of the mean lagrangian remainder
    u_sigma: n_s x 0 numpy 1darray[casadi.SX.sym]
        The upper bound of the over-approximation of the variance lagrangian remainder
    """

    n_s = q.shape[0]
    s = np.eye(n_s)
    b = mtimes(s, s.T)

    evals = matrix_norm_2_generalized(b, q)

    r_sqr = vec_max(evals)

    u_mu = l_mu * r_sqr * 0.5
    u_sigma = l_sigma * sqrt(r_sqr)

    return u_mu, u_sigma


def vec_max(x):
    """ Compute (symbolically) the maximum element in a vector

    Parameters
    ----------
    x : nx1 or array
        The symbolic input array
    """
    n, _ = x.shape

    if n == 1:
        return x[0]
    c = fmax(x[0], x[1])
    if n > 2:
        for i in range(1, n - 1):
            c = fmax(c, x[i + 1])
    return c


def matrix_norm_2_generalized(a, b_inv, x=None, n_iter=None):
    """ Get largest generalized eigenvalue of the pair inv_a^{-1},b

    get the largest eigenvalue of the generalized eigenvalue problem
        a x = \lambda b x
    <=> b x = (1/\lambda) a x

    Let \omega := 1/lambda

    We solve the problem
        b x = \omega a x
    using the inverse power iteration which converges to
    the smallest generalized eigenvalue \omega_\min

    Hence we can get  \lambda_\max = 1/\omega_\min,
        the largest eigenvalue of a x = \lambda b x

    """
    n, _ = a.shape
    if x is None:
        x = np.eye(n, 1)
        x /= norm_2(x)

    if n_iter is None:
        n_iter = 2 * n ** 2

    y = mtimes(b_inv, mtimes(a, x))
    for i in range(n_iter):
        x = y / norm_2(y)
        y = mtimes(b_inv, mtimes(a, x))

    return mtimes(y.T, x)


def dsqr_x0_2_line_x1x2(x0, x1, x2):
    assert x0.shape == x1.shape == x2.shape
    assert x0.shape[1] == 1
    # Eq 6 in https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    numerator1 = mtimes((x1-x0).T, x1-x0) * mtimes((x2-x1).T, x2-x1)
    numerator2 = (mtimes((x1-x0).T, x2-x1))**2
    denominator = mtimes((x2-x1).T, x2-x1)
    assert numerator1.shape == (1, 1) == numerator2.shape == denominator.shape
    d_sqr = (numerator1 - numerator2) / denominator
    assert d_sqr.shape == (1, 1)
    return d_sqr
