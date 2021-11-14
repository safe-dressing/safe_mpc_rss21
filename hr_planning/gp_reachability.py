# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.

Adapted from https://github.com/befelix/safe-exploration/blob/master/safe_exploration/gp_reachability.py

Created on Mon Sep 25 09:18:58 2017

@author: tkoller
'''

import numpy as np
from IPython import embed

# from casadi import SX, MX, mtimes, vertcat, sum1, sqrt, Function
# from casadi import reshape as cas_reshape

# from hr_planning.utils_casadi import compute_remainder_overapproximations
from hr_planning.utils import compute_remainder_overapproximations
from hr_planning.utils import compute_remainder_overapproximations_casadi
# from hr_planning.utils_ellipsoid_casadi import sum_two_ellipsoids, ellipsoid_from_rectangle
from hr_planning.utils_ellipsoid import sum_two_ellipsoids, ellipsoid_from_rectangle


def onestep_reachability(pR, pH, qH, beta, ssm, l_mu, l_sigma, t_z_gp=None,
                         prop_vel=False, dt=None):
    """Overapproximate the reachable set of states under the control,
    which is the robot position.

    Parameters
    ----------
    pR: n_pR x 1 ndarray[float | casadi.MX]
        Current position of the robot
    pH: n_pH x 1 ndarray[float | casadi.MX]
        Center of the overapproximated current pH (position of the human) ellipsoid
    qH: n_pH x n_pH ndarray[float | casadi.MX]
        Shape PSD matrix of the overapproximated current pH ellipsoid.
    beta: 1x1 ndarray[float | casadi.MX]
        The scaling of the semi-axes of pH's uncertainty matrix
        corresponding to a level-set of the gaussian pdf.
    ssm: SimpleGPModel
        The gp representing the human-robot dynamics
    l_mu: 1d_array of size n_pH
        Lipschitz constant of the gradient of the GP mean function
        (per pH dimension)
    l_sigma: 1d_array of size n_pH
        Lipschitz constant of the gradient of the GP stddev function
        (per pH dimension)
    t_z_gp: n_x_gp_in x n_x np.ndarray[float], optional
        = lin_trafo_gp_input in safempc_simple.py
        Allows for a linear transformation of the gp input
        (e.g. removing an input)
    prop_vel: bool,
              in addition to overapproximate pH, whether to also overapproximate vH.
    dt: sec,
        duration of one time step, used in overapproximating vH.

    Returns:
    -------
    pH_new: n_pH x 1 ndarray[float | casadi.MX]
        Center of the overapproximated next pH ellipsoid
    qH_new: n_pH x n_pH ndarray[float | casadi.MX], optional
        New shape matrix of the overapproximated next pH ellipsoid.
    """

    # Our method is described formally in Appendix C(A).
    # 1. The bound of the human position pH is same as the paper:
    # <Learning-based Model Predictive Control for Safe
    # Exploration and Reinforcement Learning>
    # Torsten Koller*, Felix Berkenkamp*, Matteo Turchetta,
    # Joschka Boedecker and Andreas Krause.
    # https://arxiv.org/pdf/1906.12189.pdf
    # 2. The bound of the human velocity vH is based on Koller's paper.
    # Hence, we use Koller's code to bound pH and adapt it a bit to bound vH.
    # All the following equation numbers refer to the equations in Koller's paper.

    n_pH = pH.shape[0]
    n_pR = pR.shape[0]
    assert pR.shape == (n_pR, 1)
    assert pH.shape == (n_pH, 1)

    # Allows for a linear transformation of the gp input
    # (e.g. removing an input)
    if t_z_gp is None:
        t_z_gp = np.eye(n_pH)

    # Equation (35) = 1st ellipsoid as (33) + 2nd ellipsoid bounding (34).
    # a. If the current pH is a point:
    # Equation (35) = a point as (33) + 2nd ellipsoid bounding (34).
    if qH is None:
        # 1. 1st ellipsoid (since qH is None, 1st ellip becomes a point)
        # (1) Uncertainty of position = equation (33)

        # Predict the mean and var of the next pH given z_bar
        x_bar = np.dot(t_z_gp, pH)
        u_bar = pR
        # zbar = 1x(npH + npR)
        z_bar = np.hstack((x_bar.T, u_bar.T))
        mu_new, pred_var = ssm(z_bar)
        mu_new = mu_new.T
        pred_var = pred_var.T
        # mu_new=npHx1, pred_var=npHx1

        # XXX: next_pH = cur_pH + gp = x_bar + mu_new
        pH_new = mu_new + x_bar

        # (2) Uncertainty of velocity
        vH_new = None
        if prop_vel:
            assert mu_new.shape == x_bar.shape
            # XXX: next_vH = (next_pH-cur_pH)/dt = gp/dt = mu_new/dt
            vH_new = mu_new / dt

        assert not (pred_var == 0).all(), "ERROR! var is all 0, plz check!"

        # 2. 2nd ellipsoid
        # (1) Uncertainty of position
        # 2.1. Equation (34):
        # qH=None => R=a point => l(R,u)=0 according to (24),
        # => RHS of (34) = \beta*\sigma(z_bar) = rkhs_bound,
        # where \sigma = sqrt(pred_var)
        # rkhs_bound = npHx1
        rkhs_bound = beta * np.sqrt(pred_var)

        # 2.2. Construct the 2nd ellipsoid to bound (34):
        # qH_new = (2,2)
        qH_new = ellipsoid_from_rectangle(rkhs_bound)
        assert not (qH_new == 0).all(), "ERROR! qH_new is all 0, plz check!"

        # (2) Uncertainty of velocity
        vqH_new = None
        if prop_vel:
            rkhs_bound_vel = rkhs_bound / dt
            vqH_new = ellipsoid_from_rectangle(rkhs_bound_vel)
            assert not (vqH_new == 0).all(), "ERROR! vH_new is all 0, plz check!"

        return pH_new, qH_new, pred_var, vH_new, vqH_new

    # b. If the current pH is an ellipsoid:
    # Equation (35) = 1st ellipsoid as (33) + 2nd ellipsoid bounding (34).
    else:
        # 1. 1st ellipsoid
        # (1) Uncertainty of position
        # 1.1. GP linearization
        # npHx1 = 2x1
        x_bar = np.dot(t_z_gp, pH)
        # npRx1 = 2x1
        u_bar = pR
        # zbar = 1x(npH + npR)
        z_bar = np.hstack((x_bar.T, u_bar.T))

        # Predict the mean and var of the next state given x_bar, u_bar.
        # mu_0=npHx1=2x1 = \mu_n(z_bar) (the paragraph above equation (29)).
        # var_0=npHx1=2x1 = variance, so it has to be sqrt'ed later.
        # jac_mu=npHx(npH+npR)=2x4 = J_{\mu_n}(z_bar) = [A_{\mu_n}, B_{\mu_n}]
        # (the paragraph above equation (29)).
        mu_0, var_0, jac_mu = ssm(z_bar, compute_gradients=True)
        mu_0 = mu_0.T
        var_0 = var_0.T
        jac_mu = jac_mu[0]
        try:
            assert not np.isnan(jac_mu).any()
        except:
            print("jac_mu is nan")
            embed()

        # a_mu = npH x npH = 2x2 = A_{\mu_n}
        # (the paragraph above equation (29)).
        # n_x_in = npH = 2
        n_x_in = np.shape(t_z_gp)[0]
        # a_mu = npH x npH = 2x2 = A_{\mu_n}
        # (the paragraph above equation (29)).
        a_mu = jac_mu[:, :n_x_in]
        a_mu = np.dot(a_mu, t_z_gp)
        # b_mu = npH x npR = 2x2 = B_{\mu_n}
        # (the paragraph above equation (29)).
        # b_mu = jac_mu[:, n_x_in:]

        # 1.2. 1st ellipsoid
        # Equation (33): our notation H is from equation (38).
        assert a_mu.shape == np.eye(n_pH).shape
        H = a_mu + np.eye(n_pH)
        qH_new_e1 = np.dot(H, np.dot(qH, H.T))
        # XXX: next_pH = cur_pH + gp = x_bar + mu_0
        pH_new_e1 = mu_0 + x_bar
        try:
            assert not np.isnan(qH_new_e1).any()
        except:
            print("qH_new_e1 is nan")
            embed()

        # (2) Uncertainty of velocity
        vH_new_e1 = None
        vqH_new_e1 = None
        if prop_vel:
            # 1.1. GP linearization
            # npHx1
            assert mu_0.shape == x_bar.shape
            # XXX: next_vH = (next_pH-cur_pH)/dt = gp/dt = mu_0/dt
            vH_new_e1 = mu_0 / dt
            # 1.2. 1st ellipsoid
            # npHxnpH
            H_for_vel = a_mu / dt
            vqH_new_e1 = np.dot(H_for_vel, np.dot(qH, H_for_vel.T))

        # 2. 2nd ellipsoid
        # (1) Uncertainty of position
        # 2.1. Equation (34):
        # ub_mean = l_mu / 2 * l^2 = npHx1
        # ub_sigma = l_sigma * l = npHx1
        ub_mean, ub_sigma = compute_remainder_overapproximations_casadi(
                qH, l_mu, l_sigma)
        ub_mean = np.reshape(ub_mean, (n_pH, 1))
        ub_sigma = np.reshape(ub_sigma, (n_pH, 1))
        # b_sigma_eps = npHx1
        b_sigma_eps = beta * (np.sqrt(var_0) + ub_sigma)

        # 2.2. Construct the 2nd ellipsoid to bound (34):
        Q_lagrange_mu = ellipsoid_from_rectangle(ub_mean)
        p_lagrange_mu = np.zeros((n_pH, 1))
        try:
            assert not np.isnan(Q_lagrange_mu).any()
        except:
            print("Q_lagrange_mu is nan")
            embed()

        Q_lagrange_sigm = ellipsoid_from_rectangle(b_sigma_eps)
        p_lagrange_sigm = np.zeros((n_pH, 1))
        try:
            assert not np.isnan(Q_lagrange_sigm).any()
        except:
            print("Q_lagrange_sigm is nan")
            embed()

        p_sum_lagrange, Q_sum_lagrange = sum_two_ellipsoids(p_lagrange_sigm,
                                                            Q_lagrange_sigm,
                                                            p_lagrange_mu,
                                                            Q_lagrange_mu)
        try:
            assert not np.isnan(Q_sum_lagrange).any()
        except:
            print("Q_sum_lagrange is nan")
            embed()

        # (2) Uncertainty of velocity
        v_sum_lagrange = None
        vQ_sum_lagrange = None
        if prop_vel:
            # 2.1. Equation (34):
            # npHx1
            ub_mean_vel = ub_mean / dt
            # npHx1
            b_sigma_eps_vel = b_sigma_eps / dt

            # 2.2. Construct the 2nd ellipsoid to bound (34):
            vQ_lagrange_mu = ellipsoid_from_rectangle(ub_mean_vel)
            v_lagrange_mu = np.zeros((n_pH, 1))
            try:
                assert not np.isnan(vQ_lagrange_mu).any()
            except:
                print("vQ_lagrange_mu is nan")
                embed()

            vQ_lagrange_sigm = ellipsoid_from_rectangle(b_sigma_eps_vel)
            v_lagrange_sigm = np.zeros((n_pH, 1))
            try:
                assert not np.isnan(vQ_lagrange_sigm).any()
            except:
                print("vQ_lagrange_sigm is nan")
                embed()

            v_sum_lagrange, vQ_sum_lagrange = sum_two_ellipsoids(
                    v_lagrange_sigm, vQ_lagrange_sigm,
                    v_lagrange_mu, vQ_lagrange_mu)
            try:
                assert not np.isnan(vQ_sum_lagrange).any()
            except:
                print("vQ_sum_lagrange is nan")
                embed()

        # 3. Combine 2 ellipsoid in equation (35):
        # (1) Uncertainty of position
        p_1, q_1 = sum_two_ellipsoids(
                p_sum_lagrange, Q_sum_lagrange, pH_new_e1, qH_new_e1)
        # (2) Uncertainty of velocity
        v_1 = None
        vq_1 = None
        if prop_vel:
            v_1, vq_1 = sum_two_ellipsoids(
                    v_sum_lagrange, vQ_sum_lagrange, vH_new_e1, vqH_new_e1)

        return p_1, q_1, var_0, v_1, vq_1


def multi_step_reachability(pH_0, qH_0, pR_0, pRs_1_T, gp, l_mu, l_sigma,
                            beta, t_z_gp=None, prop_vel=False, dt=None):
    """Generate trajectory of reachability sets
    by iteratively computing the one-step reachability.

    Parameters
    ----------
    pH_0: n_pH x 1 ndarray[float | casadi.MX]
        Initial position of the human
    qH_0: n_pH x n_pH ndarray[float | casadi.MX]
        Shape PSD matrix of the overapproximated initial pH ellipsoid.
    pR_0: n_pR x 1 ndarray[float | casadi.MX]
        Initial position of the robot
    pRs_1_T: n_safe x n_pR ndarray[float | casadi.MX]
        Positions of the robot from time 1 to T(=n_safe)
    gp: SimpleGPModel
        The gp representing the dynamics
    l_mu: 1d_array of size n_pH
        Lipschitz constant of the gradient of the GP mean function
        (per pH dimension)
    l_sigma: 1d_array of size n_pH
        Lipschitz constant of the gradient of the GP stddev function
        (per pH dimension)
    beta: 1x1 ndarray[float | casadi.MX]
        The scaling of the semi-axes of pH's uncertainty matrix
        corresponding to a level-set of the gaussian pdf.
    t_z_gp: n_x_gp_in x n_x np.ndarray[float], optional
        = lin_trafo_gp_input in safempc_simple.py
        Allows for a linear transformation of the gp input
        (e.g. removing an input)
    prop_vel: bool,
              in addition to overapproximate pH, whether to also overapproximate vH.
    dt: sec,
        duration of one time step, used in overapproximating vH.

    Returns
    -------
    pHs_1_T: T x n_pH = centers of ellipsoids for pH along the time steps.
    qHs_1_T: T x n_pH^2 = shape of ellipsoids for pH along the time steps.
    vHs_1_T: T x n_pH = centers of ellipsoids for vH along the time steps.
              (only available if prop_vel=True)
    vqHs_1_T: T x n_pH^2 = shape of ellipsoids for vH along the time steps.
              (only available if prop_vel=True)
    pH_gp_pred_sigma_1_T = (T*n_pH) x 1
           = the GP predictive stddev of pH along the time steps.
    """

    # Our method is described formally in Appendix C(B).
    # 1. The reachable sets for the human position pH is same as Sec.V(B) in paper:
    # <Learning-based Model Predictive Control for Safe
    # Exploration and Reinforcement Learning>
    # Torsten Koller*, Felix Berkenkamp*, Matteo Turchetta,
    # Joschka Boedecker and Andreas Krause.
    # https://arxiv.org/pdf/1906.12189.pdf
    # 2. The reachable sets for the human velocity vH is based on Koller's paper.
    # Hence, we use Koller's code for pH and adapt it a bit for vH.

    n_pH = pH_0.shape[0]
    n_pR = pR_0.shape[0]
    h_safe = pRs_1_T.shape[0]
    assert pRs_1_T.shape == (h_safe, n_pR)
    assert pR_0.shape == (n_pR, 1)
    assert pH_0.shape == (n_pH, 1)

    # pH_new = npH x 1 = 2x1
    # = the center of the ellipsoid for the next state.
    # qH_new = npH x npH = 2x2
    # = the shape matrix of the ellipsoid for the next state.
    # pH_gp_pred_sigma_new = npH x 1 = 2x1 = the variance of the center.
    pH_new, qH_new, pH_gp_pred_sigma_new, vH_new, vqH_new\
        = onestep_reachability(
            pR=pR_0, pH=pH_0, qH=qH_0, beta=beta,
            ssm=gp, l_mu=l_mu, l_sigma=l_sigma, t_z_gp=t_z_gp,
            prop_vel=prop_vel, dt=dt)

    # Data format to save all of these in the future steps via concat
    # pHs_1_T = T x npH
    pHs_1_T = pH_new.T
    # qHs_1_T = T x (npH^2)
    qHs_1_T = qH_new.reshape((1, n_pH * n_pH))
    # pH_gp_pred_sigma_1_T = (T*npH) x 1
    pH_gp_pred_sigma_1_T = pH_gp_pred_sigma_new

    vHs_1_T = None
    vqHs_1_T = None
    if prop_vel:
        # vHs_1_T = T x npH
        vHs_1_T = vH_new.T
        # vqHs_1_T = T x (npH^2)
        vqHs_1_T = vqH_new.reshape((1, n_pH * n_pH))

    for t_minus_1 in range(h_safe - 1):
        pH_old = pH_new
        qH_old = qH_new
        # pRs_1_T[-1, :] will not be used, since that is the end
        # of the trajectory and no more further propagation.
        pR_old = np.reshape(pRs_1_T[t_minus_1, :], (n_pR, 1))
        pH_new, qH_new, pH_gp_pred_sigma_new, vH_new, vqH_new\
            = onestep_reachability(
                pR=pR_old, pH=pH_old, qH=qH_old, beta=beta,
                ssm=gp, l_mu=l_mu, l_sigma=l_sigma, t_z_gp=t_z_gp,
                prop_vel=prop_vel, dt=dt)

        pHs_1_T = np.vstack((pHs_1_T, pH_new.T))
        qHs_1_T = np.vstack((qHs_1_T, np.reshape(qH_new, (1, n_pH * n_pH))))
        pH_gp_pred_sigma_1_T = np.vstack((
                pH_gp_pred_sigma_1_T, pH_gp_pred_sigma_new))

        if prop_vel:
            vHs_1_T = np.vstack((vHs_1_T, vH_new.T))
            vqHs_1_T = np.vstack((vqHs_1_T, np.reshape(vqH_new, (1, n_pH * n_pH))))

    assert pHs_1_T.shape == (h_safe, n_pH)
    assert qHs_1_T.shape == (h_safe, n_pH**2)
    assert pH_gp_pred_sigma_1_T.shape == (h_safe*n_pH, 1)
    if prop_vel:
        assert vHs_1_T.shape == (h_safe, n_pH)
        assert vqHs_1_T.shape == (h_safe, n_pH**2)

    return pHs_1_T, qHs_1_T, pH_gp_pred_sigma_1_T, vHs_1_T, vqHs_1_T


def lin_ellipsoid_safety_distance(p_center, q_shape, h_mat, h_vec, c_safety=1.0):
    """ Compute the distance between eLlipsoid and polytope

    Evaluate the distance of an  ellipsoid E(p_center,q_shape), to a polytopic set
    of the form:
        h_mat * x <= h_vec.

    For reference, please see Eq.41 in
    <Learning-based Model Predictive Control for Safe
    Exploration and Reinforcement Learning>
    Torsten Koller*, Felix Berkenkamp*, Matteo Turchetta,
    Joschka Boedecker and Andreas Krause.
    https://arxiv.org/pdf/1906.12189.pdf

    Parameters
    ----------
    p_center: n_s x 1 array[float]
        The center of the state ellipsoid
    q_shape: n_s x n_s array[float]
        The shape matrix of the state ellipsoid
    h_mat: m x n_s array[float]
        The shape matrix of the safe polytope (see above)
    h_vec: m x 1 array[float]
        The additive vector of the safe polytope (see above)

    Returns
    -------
    d_safety: 1darray[float] of length m
        The distance of the ellipsoid to the polytope. If d < 0 (elementwise),
        the ellipsoid is inside the poltyope (safe), otherwise safety is not guaranteed.
    """

    # m = the number of constraints
    m, n_s = np.shape(h_mat)
    assert np.shape(p_center) == (n_s, 1), "p_center has to have shape n_s x 1"
    assert np.shape(q_shape) == (n_s, n_s), "q_shape has to have shape n_s x n_s"
    assert np.shape(h_vec) == (m, 1), "h_vec has to have shape m x 1"

    d_center = np.dot(h_mat, p_center)
    d_shape = c_safety * np.sqrt(
        np.sum(np.dot(q_shape, h_mat.T) * h_mat.T, axis=0)[:, None])
    d_safety = d_center + d_shape - h_vec

    return d_safety


if __name__ == "__main__":
    p = np.array([0, 1]).reshape((2, 1))
    q = np.array([[0.5, 0.1], [1.0, 2]], dtype=np.float32)
    h_mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    h_vec = np.array([5.0, 6.0]).reshape((2, 1))
    d = lin_ellipsoid_safety_distance(
            p_center=p, q_shape=q, h_mat=h_mat, h_vec=h_vec, c_safety=2.0)
    print(d)
