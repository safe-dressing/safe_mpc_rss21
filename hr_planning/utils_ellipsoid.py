# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:34:16 2017

@author: tkoller
https://github.com/befelix/safe-exploration/blob/master/safe_exploration/utils_ellipsoid.py
"""

import itertools
import warnings

import numpy as np
import scipy.linalg as sLa
from numpy import sqrt, trace
from IPython import embed
from hr_planning.visualization.utils_visualization import plot_ellipsoid_2D
import matplotlib.pyplot as plt


def sample_inside_ellipsoid(samples, p_center, q_shape, c=1.):
    """ Check if a sample is inside a given ellipsoid

    Verify if a sample is inside the ellipsoid given by the shape matrix Q
    and center p. I.e. we check if
        (s-p).TQ^{-1}(s-p) <= c

    Args:
        samples (numpy.ndarray[float]): array of shape n_samples x n_s;
        p_center (numpy.ndarray[float]): array of shape n_s x 1;
        q_shape (numpy.ndarray[float]): array of shape n_s x n_s;
        c (float, optional): The level set of the ellipsoid ( typically 1 makes sense)
    """

    d = distance_to_center(samples, p_center, q_shape)

    inside_ellipsoid_bool = d < c

    return inside_ellipsoid_bool


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

    n_samples, n_s = samples.shape

    # 4x2
    p_center_stacked = np.zeros((n_samples, n_s))
    for i in range(n_samples):
        p_center_stacked[i, :] = p_center.T
    # 4x2
    p_centered = samples - p_center_stacked
    # 2x2
    q_inv = sLa.inv(q_shape)
    # 3x3
    prod = np.dot(p_centered, np.dot(q_inv, p_centered.T))
    d2 = np.diag(prod)

    return d2


def sum_two_ellipsoids(p_1, q_1, p_2, q_2, c=None):
    """  Sum of two ellipsoids

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

    assert not np.isnan(p_new).any()
    assert not np.isnan(q_new).any()
    return p_new, q_new


def sum_ellipsoids(p, q, l=None):
    """ Ellipsoidal overapproximation of sum of multiple ellipsoids

    Compute an ellipsoidal overapproximation of the sum
    of n individual m-dimensional ellipsoids.

    from:
        @inproceedings{kurzhanskiy2006ellipsoidal,
          title={Ellipsoidal toolbox (ET)},
          author={Kurzhanskiy, Alex A and Varaiya, Pravin},
          booktitle={Decision and Control, 2006 45th IEEE Conference on},
          pages={1498--1503},
          year={2006},
          organization={IEEE}
        }

    Parameters
    ----------
    p: n x m array[float]
        The centers of the input ellipsoids
    q: n x m x m array[float]
        The shape matrices of the input ellipsoids
    l: m x 1 array[float], optional
        A non-zero vector giving the direction along which the
        outer approximation should be tight
    Returns
    -------
    p_new: n x 1 array
        The center of the resulting ellipsoid
    q_new: n x n array
        The shape matrix of the resulting ellipsoid
    """

    if l is None:
        l = np.diag(
            q[0])  # if the matrix was diagonal, this would be an edge of the rectangle
        warnings.warn(
            "Bad heuristic for choice of l. Might have to think of better ones")
    n, m = np.shape(p)

    assert n >= 2, "Need at least two input ellipsoids"
    if n == 2:
        return sum_two_ellipsoids(p[0, :, None], q[0], p[1, :, None], q[1])

    p_new = np.sum(p, axis=0)[:, None]

    c = np.sqrt(np.dot(l.T, np.dot(q[0], l)))
    q_new = (1. / c) * q[0]
    for i in range(1, n):
        c_i = np.sqrt(np.dot(l.T, np.dot(q[i], l)))
        q_new += (1. / c_i) * q[i]
        c += c_i
    q_new *= c

    return p_new, q_new


def _get_edges_hyperrectangle(l_b, u_b, m=None):
    """ Generate set of points from box-bounds

    Given a set of lower and upper bounds l_b,u_b
    defining the Box

        B = [l_b[0],u_b[0]] x ... x [l_b[-1],u_b[-1]]

    generate a set of points P which represent the box
    and can be used to fit an ellipsoid

    Inputs:
        l_b:    list of lower bounds of intervals defining box (see above)
        u_b:    list of upper bounds of intervals defining box (see above)

    Optionals:
        m:     Number of points to compute. (m < 2^n)

    Outputs:
        P:      Matrix (k-by-n) of points obtained from the bounds

    """

    warnings.warn("We don't need this anymore! Note that this function is untested!",
                  DeprecationWarning)

    assert (len(l_b) == len(u_b))

    n = len(l_b)
    L = [None] * n

    for i in range(n):
        L[i] = [l_b[i], u_b[i]]
    result = list(itertools.product(*L))

    P = np.array(result)
    if not m is None:
        assert m <= np.pow(2, n), "Cannot extract that many points"
        P = P[:m, :]

    return P


def ellipsoid_from_rectangle(u_b):
    """ Compute ellipsoid covering box

    Given a box defined by

        B = [l_b[0],u_b[0]] x ... x [l_b[-1],u_b[-1]],
    where l_b = -u_b (element-wise),
    we compute the minimum enclosing ellipsoid in closed-form
    as the solution to a linear least squares problem.
    This can be either done by a diagonal shape matrix (axis-aligned)
    or a rotated/shifted ellipsoid

    Method is described in:
        [1] :

    TODO:   Choice of point is terrible as of now, since it contains linearly dependent
            points which are not properly handled.

    Parameters
    ----------
        u_b: np.ndarray[float], array of size n x m where m = 1
            list of length n containing upper bounds of intervals defining box (see above)
    Returns
    -------
        q: np.ndarray[float, n_dim = 2], array of size n x n
            Shape matrix of covering ellipsoid

    """
    n, m = u_b.shape
    assert m == 1, "Wrong input shape"

    d = n * u_b ** 2
    d = np.reshape(d, (n,))
    q = np.diag(d)

    return q


if __name__ == "__main__":
    # ub = np.reshape(np.array([0.1, 0.2]), (2, 1))
    ub = np.reshape(np.array([1, 2]), (2, 1))
    # Here ellipsoid_from_rectangle requires that the center is at origin.
    p = np.array([0, 0]).reshape((2, 1))
    q = ellipsoid_from_rectangle(ub)
    # x = np.array(ub)[:, None]

    u_b = np.array(ub)
    l_b = -u_b
    samples = _get_edges_hyperrectangle(l_b, u_b, m=None)
    samples = np.squeeze(samples, axis=2)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax, handle = plot_ellipsoid_2D(centroid=p, Q=q, ax=ax)

    tmp = sample_inside_ellipsoid(samples=samples, p_center=p, q_shape=q, c=1.+1e-5)
    assert tmp.all()

    for i in range(samples.shape[0]):
        pt = samples[i, :]
        handles = ax.plot(pt[0], pt[1], color='g', marker='x', alpha=1.)
    plt.show()

    q_1 = np.eye(3)
    q_2 = np.eye(3)
    q_3 = np.eye(3)
    q_4 = np.eye(3)

    q = np.empty((4, 3, 3))
    q[0] = q_1
    q[1] = q_2
    q[2] = q_3
    q[3] = q_4

    p = np.zeros((4, 3))

    print((sum_ellipsoids(p, q)))
