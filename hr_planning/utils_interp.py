# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import numpy as np
from IPython import embed
from scipy import interpolate
import matplotlib.pyplot as plt


def waypts_2_pwsplines(wp_traj, dt, degree=1, plot=False):
    """
    Convert a sequence of multi-dimensional sparse waypoints
    to a sequence of interpolated multi-dimensional waypoints via splines.

    Parameters
    ----------
    wp_traj: horizon * n_s, a sequence of waypoints.
    dt: duration of 1 time step of wp_traj.
    degree: the degree of the spline fit.
    plot: bool, whether to plot or not.

    Returns
    ----------
    fs: list with length = n_s, one spline interpolated trajectory per state dimension.
    dts: list with length = horizon, time steps throughout the trajectory.
    """

    # The degree of the spline fit.
    # It is recommended to use cubic splines.
    # Even values of k should be avoided especially with small s values.
    # 1 <= k <= 5
    assert 1 <= degree <= 5

    n_s = wp_traj.shape[1]
    # wp_traj = 0, ..., end_time, where end_time=horizon*dt.
    horizon = wp_traj.shape[0] - 1
    end_time = horizon * dt
    dts, step = np.linspace(0.0, end_time, num=horizon + 1,
                            endpoint=True, retstep=True)
    # print("horizon={}, end_time={}, dts={}, step={}".format(
        # horizon, end_time, dts, step))
    assert abs(step - dt) < 1e-5, "step={}, dt={}".format(step, dt)
    assert dts.shape[0] == wp_traj.shape[0]

    fs = []
    for i in range(n_s):
        spl = interpolate.splrep(x=dts, y=wp_traj[:, i].T, k=degree)
        fs.append(spl)
    if plot:
        dts2, _ = np.linspace(0.0, end_time, num=1000,
                              endpoint=True, retstep=True)
        fig, ax = plt.subplots()
        ax.plot(dts, wp_traj, 'o', label='data')
        pHs_spl = np.zeros((len(dts2), n_s), dtype=np.float32)
        for i in range(n_s):
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html#scipy.interpolate.splev
            # When x is not in the interval defined by the knot sequence.
            # if ext=2, raise a ValueError
            pHs_spl[:, i] = interpolate.splev(x=dts2, tck=fs[i], ext=2)
        for i in range(n_s):
            ax.plot(dts2, pHs_spl[:, i], label="pwspline")
        ax.legend(loc='upper right', ncol=2)
        plt.show()
    return fs, dts


def waypts_2_zeroOrderHold(wp_traj, dt, axis, plot=False):
    """
    Convert a sequence of one-dimensional sparse waypoints
    to a sequence of interpolated one-dimensional waypoints via zero order hold.

    Parameters
    ----------
    wp_traj: horizon * n_s, a sequence of waypoints.
    dt: duration of 1 time step of wp_traj.
    axis: the axis where interpolation happens
    plot: bool, whether to plot or not.

    Returns
    ----------
    f: scipy.interpolate.interpolate.interp1d, a one-dimensional interpolated trajectory.
    """

    n_s = wp_traj.shape[1]
    # wp_traj = 0, ..., end_time, where end_time=horizon*dt.
    horizon = wp_traj.shape[0] - 1
    end_time = horizon * dt
    dts, step = np.linspace(0.0, end_time, num=horizon + 1,
                            endpoint=True, retstep=True)
    # print("horizon={}, end_time={}, dts={}, step={}".format(
        # horizon, end_time, dts, step))
    assert abs(step - dt) < 1e-5
    assert dts.shape[0] == wp_traj.shape[0]

    f = interpolate.interp1d(x=dts, y=wp_traj, kind="zero", axis=axis)

    if plot:
        dts2, _ = np.linspace(0.0, end_time, num=1000,
                              endpoint=True, retstep=True)
        fig, ax = plt.subplots()
        ax.plot(dts, wp_traj, 'o', label='data')
        interp_traj = f(dts2)
        for i in range(n_s):
            ax.plot(dts2, interp_traj[:, i], label="pwspline")
        ax.legend(loc='upper right', ncol=2)
        plt.show()
    return f


if __name__ == "__main__":
    np.random.seed(0)
    # https://stackoverflow.com/a/2891805
    np.set_printoptions(precision=3, suppress=True)

    traj = np.array([[1, 2], [3, 12], [5, 9], [3, 4]])
    dt = 1.0
    fs, dts = waypts_2_pwsplines(wp_traj=traj, dt=dt, degree=3, plot=True)
    f2 = waypts_2_zeroOrderHold(wp_traj=traj, dt=dt, axis=0, plot=True)
