# -*- coding: utf-8 -*-
"""
Adapted from the following file, with some additional functions:
# https://github.com/befelix/safe-exploration/blob/master/safe_exploration/utils_ellipsoid_casadi.py

Created on Fri Sep 22 11:08:49 2017

@author: tkoller
"""
from casadi.tools import *
from IPython import embed
from casadi import inv, mtimes, MX, vertcat, diag, pinv


def distance_to_center(samples, p_center, q_shape):
    """ Get the distance of a set of samples to the center of the ellipsoid


    Compute the distance:
        d = (s-p).T*Q^{-1}*(s-p)
    for a set of samples

    Args:
        samples (numpy.ndarray[float]): array of shape n_samples x n_s;
        p_center (numpy.ndarray[float]): array of shape n_s x 1;
        q_shape (numpy.ndarray[float]): array of shape n_s x n_s;


    Returns:
        distance (numpy.array[float]): 1darray of length n_samples containing
                                    the distance to the center of the ellipsoid
                                    (see above)
    """

    # 3, 2
    n_samples, n_s = samples.shape

    # 3x2
    p_center_stacked = MX.zeros((n_samples, n_s))
    for i in range(n_samples):
        p_center_stacked[i, :] = p_center.T
        assert p_center_stacked[i, :].shape == p_center.T.shape
    # 3x2
    p_centered = samples - p_center_stacked

    # 2x2
    q_inv = inv(q_shape + 1e-9)

    # 3x3
    prod = mtimes(p_centered, mtimes(q_inv, p_centered.T))
    d = diag(prod)
    return d


def sum_two_ellipsoids(p_1, q_1, p_2, q_2, c=None):
    """ overapproximation of sum of two ellipsoids

    Computes the ellipsoidal overapproximation of the sum of two n-dimensional
    ellipsoids.
    from:
    "A Kurzhanski, I Valyi - Ellipsoidal Calculus for Estimation and Control"

    Parameters
    ----------
    p_1,p_2: n x 1 array
        The centers of the ellipsoids to sum
    q_1,q_2: n x n array
        The shape matrices of the two ellipsoids
    c: float, optional
        The
    Returns
    -------
    p_new: n x 1 array
        The center of the resulting ellipsoid
    q_new: n x n array
        The shape matrix of the resulting ellipsoid
    """

    # choose p s.t. the trace of the new shape matrix is minimized
    if c is None:
        c = sqrt(trace(q_1) / (trace(q_2) + 1e-9))

    p_new = p_1 + p_2
    q_new = (1 + (1. / c)) * q_1 + (1 + c) * q_2

    return p_new, q_new


def ellipsoid_from_rectangle(u_b):
    """ Compute ellipsoid covering box
    Given a box defined by
        B = [l_b[0],u_b[0]] x ... x [l_b[-1],u_b[-1]],
    where l_b = -u_b (element-wise),
    we compute the minimum enclosing axis-aligned ellipsoid in closed-form
    as the solution to a linear least squares problem.
    Method is described in:
        [1] :
    Parameters
    ----------
        u_b: n x 1 array
            array containing upper bounds of intervals defining box (see above)
    Returns
    -------
        q: np.ndarray[float, n_dim = 2], array of size n x n
            The (diagonal) shape matrix of covering ellipsoid
    """

    n, m = u_b.shape
    assert m == 1, "Wrong input shape"
    d = n * u_b ** 2
    q = diag(d)

    return q


if __name__ == "__main__":
    sample1 = MX.sym("s1", (1, 2))
    sample2 = MX.sym("s2", (1, 2))
    sample3 = MX.sym("s3", (1, 2))
    samples = vertcat(sample1, sample2, sample3)
    p_center = MX.sym("c", (2, 1))
    q_shape = MX.sym("q", (2, 2))
    distance_to_center(samples, p_center, q_shape)
