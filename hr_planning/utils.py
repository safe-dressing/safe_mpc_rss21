# -*- coding: utf-8 -*-
"""
Adapted from the following file, with some additional functions:

# https://github.com/befelix/safe-exploration/blob/master/safe_exploration/utils.py
Created on Wed Sep 20 10:43:16 2017

@author: tkoller
"""
import inspect
import functools
import warnings
import itertools

import numpy as np
import scipy.linalg as sLa
from casadi import reshape as cas_reshape
from numpy import diag, eye
from numpy.linalg import solve, norm
from numpy.matlib import repmat
from IPython import embed


def infinite_continuous_lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    http://www.mwm.im/lqr-controllers-with-python/
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # ref Bertsekas, p.151
    # first, try to solve the ricatti equation
    X = np.matrix(sLa.solve_continuous_are(A, B, Q, R))
    # compute the LQR gain
    K = np.matrix(sLa.inv(R)*(B.T*X))
    eigVals, eigVecs = sLa.eig(A-B*K)
    return K, X, eigVals


def infinite_discrete_lqr(a, b, q, r):
    """Solve the discrete time lqr controller.
    http://www.mwm.im/lqr-controllers-with-python/
    Get the feedback controls from linearized system at the current time step
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    For a discrete time system, find the infinite horizon optimal feedback
    controller to steer the system to the origin with u = -K*x.
    """
    # Solve discrete-time algebraic Riccati equation (DARE)
    x = np.matrix(sLa.solve_discrete_are(a, b, q, r))

    k = np.matrix(sLa.inv(b.T * x * b + r) * (b.T * x * a))

    eigVals, eigVecs = sLa.eig(a - b * k)

    return np.asarray(k), np.asarray(x), eigVals


def finite_discrete_lqr(A, B, Q, R, horizon):
    # http://www.cs.toronto.edu/~florian/courses/imitation_learning/lectures/Lecture2.pdf
    '''
    # Double integrator robot
    # 2D: A = [0010; 0001; 0000; 0000], b = [00; 00; 1/m 0; 1/m 0]
    tmp = self.env.n_pR + self.env.n_vR
    A = np.zeros((tmp, tmp))
    B = np.zeros((tmp, self.env.n_u))
    for i in range(self.env.n_vR):
        A[i, self.env.n_pR + i] = 1.
    for i in range(self.env.n_u):
        B[self.env.n_pR + i, i] = 1. / self.env.mR
    '''
    Ps = [None] * (horizon + 1)
    Ks = [None] * horizon
    Ps[0] = Q
    for t in range(horizon):
        # P_{n-1} = Ps[t]
        # K_{n} = Ks[t]
        BPB = np.dot(np.dot(B.T, Ps[t]), B)
        BPA = np.dot(np.dot(B.T, Ps[t]), A)
        Ks[t] = - np.dot(np.linalg.inv(R + BPB), BPA)
        KRK = np.dot(np.dot(Ks[t].T, R), Ks[t])
        A_BK = A + np.dot(B, Ks[t])
        Ps[t + 1] = Q + KRK + np.dot(np.dot(A_BK.T, Ps[t]), A_BK)
    return Ks


def rsetattr(obj, attr, val):
    """
    from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


sentinel = object()


def rgetattr(obj, attr, default=sentinel):
    """
    from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """
    if default is sentinel:
        _getattr = getattr
    else:
        def _getattr(obj, name):
            return getattr(obj, name, default)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def reshape_derivatives_3d_to_2d(derivative_3d):
    """ Reshape 3D derivative tensor to 2D derivative

    Given a function f: \R^{n_in} \to \R^{r \times s} we get a derivative tensor
    df: \R^{n_in} \to \R^{r \times s \times n_in} that needs to be reshaped to
    df_{2d}: \R^{n_in} \to \R^{r * s \times n_in} as casadi only allows for 2D arrays.

    The reshaping rule has to follow the casadi rule for these kinds of reshape operations

    TO-DO: For now this is only tested implicitly through test_state_space_model/test_ssm_evaluator_derivatives_passed_correctly
           by checking if the SSMEvaluator passes the gradients in the right format to casadi (with the use of this function)
    Parameters
    ----------
    derivative_3d: 3D-array[float]
        A (r,s,n_in) array representing the evaluated derivative tensor df

    Returns
    -------
    derivative_2d:
        A (r*s,n_in) array representing the reshaped, evaluated derivative tenor df_{2d}
    """
    r, s, n_in = np.shape(derivative_3d)

    return np.reshape(derivative_3d, (r * s, n_in))


def compute_remainder_overapproximations(q, l_mu, l_sigma):
    """ Compute the (hyper-)rectangle over-approximating the lagrangians of mu and sigma

    Parameters
    ----------
    q: n_s x n_s ndarray[float]
        The shape matrix of the current state ellipsoid
    l_mu: n x 0 numpy 1darray[float]
        The lipschitz constants for the gradients of the predictive mean
    l_sigma n x 0 numpy 1darray[float]
        The lipschitz constans on the predictive variance

    Returns
    -------
    u_mu: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the mean lagrangian remainder
    u_sigma: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the variance lagrangian remainder
    """
    n_s = q.shape[0]
    s = np.eye(n_s)
    b = np.dot(s, s.T)
    qb = np.dot(q, b)
    evals, _ = sLa.eig(qb)
    r_sqr = np.max(evals)

    u_mu = l_mu * r_sqr * 0.5
    u_sigma = l_sigma * np.sqrt(r_sqr)

    return u_mu, u_sigma


def compute_remainder_overapproximations_casadi(q, l_mu, l_sigma):
    """ Compute the (hyper-)rectangle over-approximating the lagrangians of mu and sigma
    Parameters
    ----------
    q: n_s x n_s ndarray[float]
        The shape matrix of the current state ellipsoid
    k_fb: n_u x n_s ndarray[float]
        The linear feedback term
    l_mu: n x 0 numpy 1darray[float]
        The lipschitz constants for the gradients of the predictive mean
    l_sigma n x 0 numpy 1darray[float]
        The lipschitz constans on the predictive variance
    Returns
    -------
    u_mu: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the mean lagrangian remainder
    u_sigma: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the variance lagrangian remainder
    """

    n_s = q.shape[0]
    s = np.eye(n_s)
    b = np.dot(s, s.T)

    evals = matrix_norm_2_generalized(b, q)

    r_sqr = vec_max(evals)

    u_mu = l_mu * r_sqr * 0.5
    u_sigma = l_sigma * np.sqrt(r_sqr)

    assert not np.isnan(u_mu).any()
    assert not np.isnan(u_sigma).any()
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
    c = np.maximum(x[0], x[1])
    if n > 2:
        for i in range(1, n - 1):
            c = np.maximum(c, x[i + 1])
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
        x /= np.linalg.norm(x, ord=2)

    if n_iter is None:
        n_iter = 2 * n ** 2

    y = np.dot(b_inv, np.dot(a, x))
    for i in range(n_iter):
        x = y / np.linalg.norm(y, ord=2)
        y = np.dot(b_inv, np.dot(a, x))

    assert not np.isnan(y).any()
    assert not np.isnan(x).any()
    return np.dot(y.T, x)


def dsqr_x0_2_line_x1x2(x0, x1, x2):
    assert x0.shape == x1.shape == x2.shape
    assert x0.shape[1] == 1
    # Eq 6 in https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    numerator1 = np.dot((x1-x0).T, x1-x0) * np.dot((x2-x1).T, x2-x1)
    numerator2 = (np.dot((x1-x0).T, x2-x1))**2
    denominator = np.dot((x2-x1).T, x2-x1)
    assert numerator1.shape == (1, 1) == numerator2.shape == denominator.shape
    d_sqr = (numerator1 - numerator2) / denominator
    assert d_sqr.shape == (1, 1)
    return d_sqr


if __name__ == "__main__":
    x0 = np.array([5, 0])
    x1 = np.array([1, 2])
    x2 = np.array([3, 2])
    x0 = x0.reshape(2, 1)
    x1 = x1.reshape(2, 1)
    x2 = x2.reshape(2, 1)
    dqrt = dsqr_x0_2_line_x1x2(x0, x1, x2)
    assert dqrt.shape == (1, 1)
    assert dqrt == 2**2

    x0 = np.array([3, 2])
    x1 = np.array([5, 0])
    x2 = np.array([1, 2])
    x0 = x0.reshape(2, 1)
    x1 = x1.reshape(2, 1)
    x2 = x2.reshape(2, 1)
    dqrt = dsqr_x0_2_line_x1x2(x0, x1, x2)
    assert dqrt.shape == (1, 1)
    assert dqrt == 0.8
