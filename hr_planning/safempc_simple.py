# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.

Adapted from https://github.com/befelix/safe-exploration/blob/master/safe_exploration/safempc_simple.py

Created on Mon Sep 25 09:18:58 2017

@author: tkoller
'''

import casadi as cas
import numpy as np
from IPython import embed
from casadi import MX, mtimes, vertcat, sum2, sqrt, jacobian
from casadi import reshape as cas_reshape
from casadi import exp as cas_exp
from casadi import Importer, external
import os
from scipy import interpolate
import time
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

import hr_planning
from hr_planning.visualization.utils_visualization import print_FAIL, print_OK
from hr_planning.gp_reachability_casadi import multi_step_reachability
from hr_planning.gp_reachability import multi_step_reachability as multi_step_reachability_value
from hr_planning.utils_ellipsoid_casadi import distance_to_center as distance_to_center_casadi
from hr_planning.gp_reachability_casadi import lin_ellipsoid_safety_distance
from hr_planning.utils_ellipsoid import distance_to_center
from hr_planning.ssm_gpy.gaussian_process import SimpleGPModel
from hr_planning.utils_interp import waypts_2_pwsplines
from hr_planning.utils_casadi import dsqr_x0_2_line_x1x2

dressing_2d_dist_pR_2_line_pH_pH_shoulder_in_cost = False


class SimpleSafeMPC:
    def __init__(self, env, ssm, config_path_rmpc, result_dir):
        """
        Class for an MPC to compute robot trajectory to ensure human physical safety.

        Parameters
        ----------
        env: HREnv class
        ssm: SimpleGPModel
            The gp representing the human-robot dynamics
        config_path_rmpc: path to the configuration file of the robot MPC.
        result_dir: path to save results.
        """

        self.env = env

        with open(config_path_rmpc) as f:
            self.config_r_mpc = yaml.load(f, Loader=yaml.FullLoader)

        # h_safe = Length of the safety trajectory (number of safe controls)
        self.h_safe = self.config_r_mpc["h_safe"]

        w_u = self.config_r_mpc["w_u"]
        assert len(w_u) == self.env.n_uR
        self.w_u = np.diag(w_u)
        w_goal = self.config_r_mpc["w_goal"]
        assert len(w_goal) == self.env.n_pR
        self.w_goal = np.diag(w_goal)
        self.w_dist_2_pH = float(self.config_r_mpc["w_dist_2_pH"])
        if "w_dist_2_line_pH_pH_shoulder" in self.config_r_mpc:
            self.w_dist_2_line_pH_pH_shoulder = float(
                    self.config_r_mpc["w_dist_2_line_pH_pH_shoulder"])

        # Lipschitz constant of the gradient of the GP mean function
        # (per pH dimension)
        self.l_mu = np.array(self.config_r_mpc["l_mu"])
        assert self.l_mu.shape[0] == self.env.n_pH

        # Lipschitz constant of the gradient of the GP stddev function
        # (per pH dimension)
        self.l_sigma = np.array(self.config_r_mpc["l_sigma"])
        assert self.l_sigma.shape[0] == self.env.n_pH

        # Duration of one time step, used in overapproximating vH (sec).
        self.dt = self.env.dt_Rmpc

        self.ssm = ssm
        assert self.ssm.num_states == self.ssm.n_s_out == self.env.n_vH\
            == self.ssm.n_s_in == self.env.n_pH
        assert self.ssm.num_actions == self.ssm.n_u == self.env.n_pR
        self.ssm_forward = ssm.get_forward_model_casadi(True)

        # Allows for a linear transformation of the gp input
        # (e.g. removing an input).
        self.lin_trafo_gp_input = None
        if self.lin_trafo_gp_input is None:
            self.lin_trafo_gp_input = np.eye(self.ssm.n_s_in)

        # Receding horizon to reuse previous trajectory (Alg.1 line 5)
        self.shifted_solution_as_init = True
        if self.shifted_solution_as_init:
            assert self.h_safe > 1, "cannot shift solution since h_safe<=1"
        # How many steps to shift
        self.shift_num = 1

        # Safe recovery policy (Assumption 2)
        self.safe_policy = lambda pR, vR, pH, vH:\
            np.zeros((self.env.n_uR), dtype=np.float32)

        self.result_dir = result_dir

        self.feas_tol = 1e-6

        self.status_feasible_MPC = 0
        self.status_infeasible_MPC_use_old_traj = 1
        self.status_infeasible_MPC_use_safe_policy = 2

        self.init_pR_pH_qH_dist_cas_fcn()
        self.init_lin_ellipsoid_safety_distance_cas_fcn()
        self.init_dsqr_x0_2_line_x1x2_cas_fcn()

        # For logging
        self.itr_idx = -1

        self.pR_mode = None
        self.task = None

        # Initialize s.t. there is no backup strategy
        self.n_fail = self.h_safe

        self.fixed_beta = 2.

        # Surrogate constraint (Paragraph below Eq.4 and Fig.3 in the paper)
        # C_{SI,i} <= max(theta1*C_{CA}, theta2*C_{CA})
        eta = MX.sym('eta', (1, 1))
        alpha = MX.sym('alpha', (1, 1))
        self.zeta_a = 0.01
        self.zeta_b = 1000.
        zeta = eta - cas.fmax(self.zeta_a*alpha, self.zeta_b*alpha)
        self.compute_zeta = cas.Function("compute_zeta", [eta, alpha], [zeta])

        # We allow user to constrain the robot states along some time indices.
        # E.g., self.t_2_constrained_pR = {0:0, 1:1} means that
        # at time index 0, pR=0;
        # at time index 1, pR=1;
        # Hence, our optimization program will NOT employ boundary constraints
        # nor dynamics constraints to states (pR,vR,u) that are related to
        # these user-specified pR's.
        self.t_2_constrained_pR = None

    def reset(self):
        self.n_fail = self.h_safe

    def check_shapes(self, pRs_1_T=None, vRs_1_T=None, us_1_T=None,
                     pHs_1_T=None, qHs_1_T=None,
                     pR_0=None, vR_0=None, pH_0=None):
        """Check matrix' shape"""
        if pRs_1_T is not None:
            assert pRs_1_T.shape == (self.h_safe, self.env.n_pR)
        if vRs_1_T is not None:
            assert vRs_1_T.shape == (self.h_safe, self.env.n_vR)
        if us_1_T is not None:
            assert us_1_T.shape == (self.h_safe, self.env.n_uR)
        if pR_0 is not None:
            assert pR_0.shape == (self.env.n_pR, 1)
        if vR_0 is not None:
            assert vR_0.shape == (self.env.n_vR, 1)
        if pH_0 is not None:
            assert pH_0.shape == (self.env.n_pH, 1)
        if pHs_1_T is not None:
            assert pHs_1_T.shape == (self.h_safe, self.env.n_pH)
        if qHs_1_T is not None:
            assert qHs_1_T.shape == (self.h_safe, self.env.n_pH**2)

    def init_pR_pH_qH_dist_cas_fcn(self):
        """Initialize casadi function that can compute
        the distance of pR, relative to the ellipsoid (pH,qH)"""
        pR = MX.sym('pR', (self.env.n_pR, 1))
        pH = MX.sym('pH', (self.env.n_pH, 1))
        qH = MX.sym('qH', (self.env.n_pH**2, 1))
        pR_pH_qH_dist = self.compute_pR_pH_qH_dist_casadi(pR, pH, qH)
        self.compute_pR_pH_qH_dist_cas_fcn = cas.Function(
                "pR_pH_qH_dist",
                [pR, pH, qH], [pR_pH_qH_dist])
        # Jacobian functions for debugging
        # https://groups.google.com/g/casadi-users/c/9ZuaCBGMa8U
        # https://groups.google.com/g/casadi-users/c/zNwlmUpLbZc
        dpR = jacobian(pR_pH_qH_dist, pR)
        dpH = jacobian(pR_pH_qH_dist, pH)
        dqH = jacobian(pR_pH_qH_dist, qH)
        self.compute_Jac_4_pR_pH_qH_dist_cas_fcn = cas.Function(
                "Jac_4_pR_pH_qH_dist_cas_fcn",
                [pR, pH, qH], [dpR, dpH, dqH])

    def init_lin_ellipsoid_safety_distance_cas_fcn(self):
        """Initialize casadi function that can compute
        the distance of ellipsoid to polytope, which is used for checking
        whether vR results in safe impact with the ellipsoid (vH,vqH)."""
        vR = MX.sym("vR", (self.env.n_vR, 1))
        vH = MX.sym("vH", (self.env.n_pH, 1))
        vqH = MX.sym("vqH", (self.env.n_pH, self.env.n_pH))
        # (2*nvR)xnvR = 4x2
        h_mat = self.env.h_mat_safe_imp_pot
        assert h_mat.shape == (self.env.n_vR*2, self.env.n_vR)
        # (2*nvR)x1 = 4x1
        tmp = self.env.h_safe_imp_pot
        # (2*nvR)x1 = 4x1
        tmp2 = vertcat(-vR, vR)
        assert tmp.shape == tmp2.shape
        h_vec = tmp + tmp2
        assert h_vec.shape == (self.env.n_vR*2, 1)
        h_ellip_dist = lin_ellipsoid_safety_distance(
                p_center=vH, q_shape=vqH, h_mat=h_mat, h_vec=h_vec)
        self.compute_lin_ellipsoid_safety_distance_cas_fcn\
            = cas.Function("lin_ellipsoid_safety_distance",
                           [vR, vH, vqH], [h_ellip_dist])

    def init_dsqr_x0_2_line_x1x2_cas_fcn(self):
        """Initialize casadi function that can compute
        the distance of a point x0, to the line segment between x1,x2.
        This is used for dressing 2D."""
        x0 = MX.sym("x0", (self.env.n_pR, 1))
        x1 = MX.sym("x1", (self.env.n_pH, 1))
        x2 = MX.sym("x2", (self.env.n_pH, 1))
        d_sqr = dsqr_x0_2_line_x1x2(x0=x0, x1=x1, x2=x2)
        self.compute_dsqr_x0_2_line_x1x2_cas_fcn\
            = cas.Function("dsqr_x0_2_line_x1x2", [x0, x1, x2], [d_sqr])

    def set_task(self, task, pR_mode=None):
        """Set the current task that the MPC is planning for."""
        if pR_mode is not None:
            self.pR_mode = pR_mode
            assert pR_mode in ["CA", "CA_SI"]
        self.task = task
        assert self.task in ["coll_avoid", "handover", "dressing_2d"]

    def init_solver(self):
        """Create an instance of a nonlinear program for ipopt solver."""

        # Decision variables
        # Horizon T = self.h_safe
        # Robot positions
        pRs_1_T = MX.sym("pRs_1_T", (self.h_safe, self.env.n_pR))
        # Robot velocities
        vRs_1_T = MX.sym("vRs_1_T", (self.h_safe, self.env.n_vR))
        # Robot controls
        us_1_T = MX.sym("us_1_T", (self.h_safe, self.env.n_uR))

        # Parameters
        # Initial robot position
        pR_0 = MX.sym("pR_0", (self.env.n_pR, 1))
        # Initial robot velocity
        vR_0 = MX.sym("vR_0", (self.env.n_vR, 1))
        # Initial human position
        pH_0 = MX.sym("pH_0", (self.env.n_pH, 1))
        # Duration of a time step
        dt = MX.sym("dt", (1, 1))

        # The scaling of the semi-axes of pH's uncertainty matrix
        # corresponding to a level-set of the gaussian pdf.
        beta = MX.sym("beta", (1, 1))

        # Constraints
        g = []
        lbg = []
        ubg = []
        g_name = []

        self._f_multistep_eval = None
        self._f_multistep_vel_eval = None

        # pHs_1_T = T x n_pH = centers of ellipsoid for pH along the time steps.
        pHs_1_T = None
        # qHs_1_T = T x n_pH^2 = shape of ellipsoid for pH along the time steps.
        qHs_1_T = None
        # sH_gp_pred_sigma_1_T = (T x n_pH) x 1
        # = the GP predictive stddev of pH along the time steps.
        sH_gp_pred_sigma_1_T = None
        # vHs_1_T = T x n_pH = centers of ellipsoid for vH along the time steps.
        vHs_1_T = None
        # vqHs_1_T = T x n_pH^2 = shape of ellipsoid for vH along the time steps.
        vqHs_1_T = None

        # (1) Propagate pos using beta
        if self.pR_mode == "CA":
            pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T, _1, _2\
                = multi_step_reachability(
                    pH_0=pH_0, qH_0=None, pR_0=pR_0, pRs_1_T=pRs_1_T,
                    beta=beta, gp=self.ssm_forward,
                    l_mu=self.l_mu, l_sigma=self.l_sigma,
                    t_z_gp=None, prop_vel=False, dt=None)
            assert _1 is None
            assert _2 is None
            assert sH_gp_pred_sigma_1_T.shape == (self.h_safe*self.env.n_pH, 1)
            # Generate open_loop trajectory function
            # [vertcat(x_0,u_0,u_{1:T},beta)],[f_x])
            self._f_multistep_eval = cas.Function(
                    "safe_multistep", [pH_0, pR_0, pRs_1_T, beta],
                    [pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T])

        # (2) Propagate vel using beta
        elif self.pR_mode == "CA_SI":
            pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T,\
                vHs_1_T, vqHs_1_T = multi_step_reachability(
                    pH_0=pH_0, qH_0=None, pR_0=pR_0, pRs_1_T=pRs_1_T,
                    beta=beta, gp=self.ssm_forward,
                    l_mu=self.l_mu, l_sigma=self.l_sigma,
                    t_z_gp=None, prop_vel=True, dt=dt)
            assert sH_gp_pred_sigma_1_T.shape == (self.h_safe*self.env.n_pH, 1)
            self._f_multistep_eval = cas.Function(
                    "safe_multistep", [pH_0, pR_0, pRs_1_T, beta],
                    [pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T])
            self._f_multistep_vel_eval = cas.Function(
                    "safe_multistep_vel", [pH_0, pR_0, pRs_1_T, beta, dt],
                    [pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T,
                        vHs_1_T, vqHs_1_T])
        else:
            raise ValueError()

        self.check_shapes(pRs_1_T=pRs_1_T, vRs_1_T=vRs_1_T, us_1_T=us_1_T,
                          pHs_1_T=pHs_1_T, qHs_1_T=qHs_1_T,
                          pR_0=pR_0, vR_0=vR_0, pH_0=pH_0)
        if self.pR_mode == "CA":
            assert vHs_1_T is None
            assert vqHs_1_T is None
        elif self.pR_mode == "CA_SI":
            self.check_shapes(pHs_1_T=vHs_1_T, qHs_1_T=vqHs_1_T)
            assert self._f_multistep_vel_eval is not None
        assert self._f_multistep_eval is not None

        interp_2_pHs_1_T = None
        if self.task == "dressing_2d":
            interp_2_pHs_1_T = self.interp_ellipsoids(pHs_1_T)

        g_bd, lbg_bd, ubg_bd, g_names_bd = self.generate_boundary_csts_bounds(
                pRs_1_T, vRs_1_T, us_1_T, terminal_goal_cst=False)
        g = vertcat(g, g_bd)
        lbg += lbg_bd
        ubg += ubg_bd
        g_name += g_names_bd

        g_dyn, lbg_dyn, ubg_dyn, g_names_dyn = self.generate_robot_dyn_csts(
                pRs_1_T, vRs_1_T, us_1_T, pR_0, vR_0, dt)
        g = vertcat(g, g_dyn)
        lbg += lbg_dyn
        ubg += ubg_dyn
        g_name += g_names_dyn

        g_workspace, lbg_workspace, ubg_workspace, g_names_workspace\
            = self.generate_workspace_csts(pHs_1_T=pHs_1_T, qHs_1_T=qHs_1_T)
        g = vertcat(g, g_workspace)
        lbg += lbg_workspace
        ubg += ubg_workspace
        g_name += g_names_workspace

        if self.pR_mode == "CA_SI":
            g_safe_vel, lbg_safe_vel, ubg_safe_vel, g_names_safe_vel\
                = self.generate_coll_avoid_safe_impact_csts(
                    pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T, qHs_1_T=qHs_1_T,
                    vRs_1_T=vRs_1_T, vHs_1_T=vHs_1_T, vqHs_1_T=vqHs_1_T)
            g = vertcat(g, g_safe_vel)
            lbg += lbg_safe_vel
            ubg += ubg_safe_vel
            g_name += g_names_safe_vel
        elif self.pR_mode == "CA":
            g_collavoid, lbg_collavoid, ubg_collavoid, g_names_collavoid\
                = self.generate_coll_avoid_csts(pHs_1_T, qHs_1_T, pRs_1_T)
            g = vertcat(g, g_collavoid)
            lbg += lbg_collavoid
            ubg += ubg_collavoid
            g_name += g_names_collavoid
        else:
            raise ValueError()

        if self.task == "dressing_2d":
            # Max dist to the pH - pH_shoulder line
            if not dressing_2d_dist_pR_2_line_pH_pH_shoulder_in_cost:
                g_d2arm, lbg_d2arm, ubg_d2arm, g_names_d2arm\
                    = self.generate_d2arm_csts(pHs_1_T, pRs_1_T)
                g = vertcat(g, g_d2arm)
                lbg += lbg_d2arm
                ubg += ubg_d2arm
                g_name += g_names_d2arm

            # Collision avoidance for the interpolated ellipsoids
            for k, pHs_1_T_interp in interp_2_pHs_1_T.items():
                if self.pR_mode == "CA_SI":
                    g_ca_interp, lbg_ca_interp, ubg_ca_interp, g_names_ca_interp\
                        = self.generate_coll_avoid_safe_impact_csts(
                            pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T_interp,
                            qHs_1_T=qHs_1_T,
                            vRs_1_T=vRs_1_T, vHs_1_T=vHs_1_T,
                            vqHs_1_T=vqHs_1_T)
                elif self.pR_mode == "CA":
                    g_ca_interp, lbg_ca_interp, ubg_ca_interp, g_names_ca_interp\
                        = self.generate_coll_avoid_csts(
                                pHs_1_T_interp, qHs_1_T, pRs_1_T)
                g = vertcat(g, g_ca_interp)
                lbg += lbg_ca_interp
                ubg += ubg_ca_interp
                g_name += g_names_ca_interp

        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)

        cost = self.generate_control_cost(us_1_T)
        if self.task == "handover":
            cost += self.generate_approach_pH_cost(pRs_1_T, pHs_1_T)
        elif self.task == "coll_avoid":
            cost += self.generate_terminal_goal_cost(pRs_1_T)
        elif self.task == "dressing_2d":
            cost += self.generate_terminal_goal_cost(pRs_1_T)
            if dressing_2d_dist_pR_2_line_pH_pH_shoulder_in_cost:
                cost += self.generate_d2arm_cost(pHs_1_T, pRs_1_T)
        else:
            raise ValueError()
        assert cost.shape == (1, 1)

        # All variables and parameters
        opt_vars = vertcat(
                pRs_1_T.reshape((-1, 1)),
                vRs_1_T.reshape((-1, 1)),
                us_1_T.reshape((-1, 1)))
        opt_params = vertcat(pR_0, vR_0, pH_0, beta, dt)
        prob = {'f': cost, 'x': opt_vars, 'p': opt_params, 'g': g}
        # Option to suppress print in IPOPT:
        # https://groups.google.com/g/casadi-users/c/a9OGbF7R4ss?pli=1
        opt = {'error_on_fail': False,
               "print_time": 0,
               'ipopt': {
                   # 'hessian_approximation': 'limited-memory',
                   "max_iter": 2000,
                   "print_level": 0,
                   "expect_infeasible_problem": "no",
                   "acceptable_tol": 1e-4,
                   "acceptable_constr_viol_tol": 1e-5,
                   "bound_frac": 0.5,
                   "start_with_resto": "no",
                   "required_infeasibility_reduction": 0.85,
                   "acceptable_iter": 8}}

        solver = cas.nlpsol('solver', 'ipopt', prob, opt)
        self.solver = solver
        self.lbg = lbg
        self.ubg = ubg
        self.g = g
        self.g_name = g_name
        self.prob = prob

    def generate_workspace_csts(self, pHs_1_T, qHs_1_T):
        """
        Formulate symbolic constraint for the workspace
        \mathcal{P}_R, \mathcal{P}_H (Sec.2) as boundary conditions,

        Parameters
        ----------
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.
        qHs_1_T: horizon x n_pH^2 = shape of ellipsoids for pH along the time steps.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pHs_1_T=pHs_1_T, qHs_1_T=qHs_1_T)
        g = []
        lbg = []
        ubg = []
        g_name = []

        n_csts = self.env.n_pH * 2
        for t in range(self.h_safe):
            # 2x1
            pH = pHs_1_T[t, :].T
            # 4x1
            qH = qHs_1_T[t, :].T
            # 2x2
            qH = cas_reshape(qH, (self.env.n_pH, self.env.n_pH))

            p_center = pH
            q_shape = qH

            # 4x2
            h_mat = np.vstack((np.eye(self.env.n_pH), -np.eye(self.env.n_pH)))
            # 4x1
            pH_max = self.env.pH_max_workspace
            pH_min = self.env.pH_min_workspace
            h_vec = np.vstack((pH_max, -pH_min))

            assert h_mat.shape == (n_csts, self.env.n_pH)
            assert h_vec.shape == (n_csts, 1)

            # dist_2_polytope = distance of ellipsoid to polytope.
            # If dist_2_polytope < 0 (elementwise),
            # the ellipsoid is inside poltyope (safe),
            # Otherwise safety is not guaranteed.
            # We want dist_2_polytope = Hi*p + \sqrt(Hi*Qt*Hi) - hi <= 0.
            # 4x1
            dist_2_polytope = lin_ellipsoid_safety_distance(
                    p_center=p_center, q_shape=q_shape,
                    h_mat=h_mat, h_vec=h_vec)
            assert dist_2_polytope.shape == (n_csts, 1)
            g = vertcat(g, dist_2_polytope)
            lbg += [-cas.inf] * n_csts
            ubg += [0.] * n_csts
            for i in range(n_csts):
                g_name += ["workspace_cst_t={}_i={}".format(t, i)]
        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_coll_avoid_safe_impact_csts(
            self, pRs_1_T, pHs_1_T, qHs_1_T,
            vRs_1_T, vHs_1_T, vqHs_1_T):
        """
        Formulate symbolic constraint for safety = collision avoidance or safe impact.

        Parameters
        ----------
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.
        qHs_1_T: horizon x n_pH^2 = shape of ellipsoids for pH along the time steps.
        vRs_1_T: horizon x n_pR = robot velocities, vR, along the time steps.
        vHs_1_T: horizon x n_pH = centers of ellipsoids for vH along the time steps.
        vqHs_1_T: horizon x n_pH^2 = shape of ellipsoids for vH along the time steps.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T,
                          qHs_1_T=qHs_1_T, vRs_1_T=vRs_1_T)
        self.check_shapes(pHs_1_T=vHs_1_T, qHs_1_T=vqHs_1_T)
        # Number of collision avoidance constraints:
        # n_ca_csts x n_pR = 3x2
        # XXX: If only check collision to 1 pt = pR, then n_ca_csts=1.
        n_ca_csts = self.env.R_col_volume_offsets.shape[0]
        assert n_ca_csts > 0
        assert self.env.R_col_volume_offsets.shape[1] == self.env.n_pR

        g = []
        lbg = []
        ubg = []
        g_name = []

        for t in range(self.h_safe):
            # 2x1
            pR = pRs_1_T[t, :].T
            # 2x1
            pH = pHs_1_T[t, :].T
            # 4x1
            qH = qHs_1_T[t, :].T
            # 2x1
            vR = vRs_1_T[t, :].T
            # 2x1
            vH = vHs_1_T[t, :].T
            # 4x1
            vqH = vqHs_1_T[t, :].T
            vqH = cas_reshape(vqH, (self.env.n_pH, self.env.n_pH))

            # Collision avoidance:
            # A pt is inside_ellipsoid if dist_2_pt < c = 1 (elemenetwise).
            # For collision avoidance, we need each dist_2_pt>1 (elemenetwise).
            # n_ca_csts x 1 = 3x1
            dist_2_pt = self.compute_pR_pH_qH_dist_casadi(pR=pR, pH=pH, qH=qH)
            # We need alphas > 0 (elementwise).
            # n_ca_csts x 1 = 3x1
            one_vec = np.ones(dist_2_pt.shape, dtype=np.float32)
            assert dist_2_pt.shape == one_vec.shape
            alphas = dist_2_pt - one_vec

            # Safe impact:
            p_center = vH
            q_shape = vqH
            # (2*nvR)xnvR = 4x2
            h_mat = self.env.h_mat_safe_imp_pot
            n_cst = h_mat.shape[0]
            assert h_mat.shape == (n_cst, self.env.n_vR)
            # (2*nvR)x1 = 4x1
            tmp = self.env.h_safe_imp_pot
            # (2*nvR)x1 = 4x1
            tmp2 = vertcat(-vR, vR)
            h_vec = tmp + tmp2
            assert h_vec.shape == (n_cst, 1)

            # Paper Eq.12:
            # dist_2_polytope = distance of ellipsoid to polytope.
            # If dist_2_polytope < 0 (elementwise),
            # the ellipsoid is inside poltyope (safe),
            # Otherwise safety is not guaranteed.
            # We want dist_2_polytope = Hi*p + \sqrt(Hi*Qt*Hi) - hi <= 0.
            dist_2_polytope = lin_ellipsoid_safety_distance(
                    p_center=p_center, q_shape=q_shape,
                    h_mat=h_mat, h_vec=h_vec)
            assert dist_2_polytope.shape == (n_cst, 1)
            # We need etas <= 0 (elementwise).
            etas = dist_2_polytope

            n_ca_csts = alphas.shape[0]
            n_si_csts = etas.shape[0]
            for alpha_i in range(n_ca_csts):
                for eta_i in range(n_si_csts):
                    alpha = alphas[alpha_i, 0]
                    eta = etas[eta_i, 0]
                    # 1x1
                    zeta = self.compute_zeta(eta, alpha)
                    # If only care about vR-vH:
                    # zeta = eta
                    # If only care about collision:
                    # zeta = -alpha

                    assert zeta.shape == (1, 1)
                    # We want zeta < 0
                    g = vertcat(g, zeta)
                    lbg += [-cas.inf]
                    ubg += [0.]
                    g_name += ["CA_SI_t={}_alpha_i={}_eta_i={}".format(
                        t, alpha_i, eta_i)]
        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_d2arm_csts(self, pHs_1_T, pRs_1_T):
        """
        Formulate symbolic constraint for arm hole constraint:
        the robot and the human arm cannot stay too far due to the limited
        size of the armhold of the clothes.

        Formally, the distance bw pR, and the line(pH, pH_shoulder) < d_max.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T)

        g = []
        lbg = []
        ubg = []
        g_name = []

        for t in range(self.h_safe):
            # 2x1
            pR = pRs_1_T[t, :].T
            # 2x1
            pH = pHs_1_T[t, :].T
            pH_sd = np.reshape(self.env.pH_shoulder, pH.shape)
            x0 = pR
            x1 = pH
            x2 = pH_sd
            d_sqr = dsqr_x0_2_line_x1x2(x0=x0, x1=x1, x2=x2)
            assert d_sqr.shape == (1, 1)

            g = vertcat(g, d_sqr)
            lbg += [-cas.inf]
            ubg += [self.env.max_dist_bw_pR_arm**2-self.feas_tol]
            g_name += ["pR_2_line_pH_pH_shoulder_i={}".format(t)]
        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_coll_avoid_csts(self, pHs_1_T, qHs_1_T, pRs_1_T):
        """
        Formulate symbolic constraint for safety = collision avoidance only.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.
        qHs_1_T: horizon x n_pH^2 = shape of ellipsoids for pH along the time steps.
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T, qHs_1_T=qHs_1_T)

        g = []
        lbg = []
        ubg = []
        g_name = []

        # Number of collision avoidance constraints:
        # n_ca_csts x n_pR = 3x2
        # XXX: If only check collision to 1 pt = pR, then n_ca_csts=1.
        n_ca_csts = self.env.R_col_volume_offsets.shape[0]
        assert n_ca_csts > 0
        assert self.env.R_col_volume_offsets.shape[1] == self.env.n_pR

        for t in range(self.h_safe):
            # 2x1
            pR = pRs_1_T[t, :].T
            # 2x1
            pH = pHs_1_T[t, :].T
            # 4x1
            qH = qHs_1_T[t, :].T
            d = self.compute_pR_pH_qH_dist_casadi(pR=pR, pH=pH, qH=qH)
            g = vertcat(g, d)
            # A pt is inside_ellipsoid if d < c = 1.
            # For collision avoidance, we need d>1.
            lbg += [1. + self.feas_tol] * n_ca_csts
            ubg += [cas.inf] * n_ca_csts
            for j in range(n_ca_csts):
                g_name += ["pR_pH_d_i={}_dim={}".format(t, j)]

        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def compute_pR_pH_qH_dist_casadi(self, pR, pH, qH):
        """In casadi, compute the distance that indicates whether
        pR is inside the ellipsoid (pH,qH) or not."""
        # n_pts x n_pR = 3x2
        n_pts = self.env.R_col_volume_offsets.shape[0]
        assert n_pts > 0
        assert self.env.R_col_volume_offsets.shape[1] == self.env.n_pR
        assert pR.shape == (self.env.n_pR, 1)
        assert pH.shape == (self.env.n_pH, 1)
        assert qH.shape == (self.env.n_pH**2, 1)

        # nsHxnsH = 2x2
        qH = cas_reshape(qH, (self.env.n_pH, self.env.n_pH))

        # n_pts x npR = 3x2
        coll_pts = MX.zeros((n_pts, self.env.n_pR))
        for j in range(n_pts):
            offset = self.env.R_col_volume_offsets[j, :]
            offset = np.reshape(offset, (self.env.n_pR, 1))
            assert pR.shape == offset.shape
            tmp = cas_reshape(pR + offset, (coll_pts[j, :].shape))
            assert tmp.shape == coll_pts[j, :].shape
            coll_pts[j, :] = cas_reshape(pR + offset, (coll_pts[j, :].shape))
        # n_pts x 1 = 3x1
        d = distance_to_center_casadi(samples=coll_pts, p_center=pH, q_shape=qH)
        return d

    def generate_boundary_csts_bounds(
            self, pRs_1_T, vRs_1_T, us_1_T, terminal_goal_cst=False,
            enforce_t_2_constrained_pR_if_available=True):
        """
        Formulate symbolic constraint for boundary for pR,vR,u.

        Parameters
        ----------
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.
        vRs_1_T: horizon x n_pR = robot velocities, vR, along the time steps.
        us_1_T: horizon x n_u = robot controls, u, along the time steps.
        terminal_goal_cst: bool, whether or not we constrain the terminal
                           state close to the goal.
        enforce_t_2_constrained_pR_if_available: bool,
            enforce_t_2_constrained_pR_if_available = True:
            if self.t_2_constrained_pR is defined,
            then this function will ignore the boundary constraints for the pR's
            that have been specified by the user in self.t_2_constrained_pR.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, vRs_1_T=vRs_1_T, us_1_T=us_1_T)
        g = []
        lbg = []
        ubg = []
        g_name = []

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # For enforce_t_2_constrained_pR_if_available:
        # If we have specified self.t_2_constrained_pR,
        # then we will ignore the constraints for the pRs that are
        # constrained by self.t_2_constrained_pR.

        # Suppose h_safe = 4.
        # pR0, [pR1, pR2, pR3, pR4] where [] = pRs_1_T
        # vR0, [vR1, vR2, vR3, vR4] where [] = pRs_1_T
        # vR0 ~ ()
        # vR1 ~ (pR0, pR1)
        # vR2 ~ (pR1, pR2)
        # vR3 ~ (pR2, pR3)
        # vR4 ~ (pR3, pR4)
        # [uR1, uR2, uR3, uR4] where [] = us_1_T
        # uR1 ~ (vR0, vR1) ~ (pR0, pR1)
        # uR2 ~ (vR1, vR2) ~ (pR0, pR1, pR2)
        # uR3 ~ (vR2, vR3) ~ (pR1, pR2, pR3)
        # uR4 ~ (vR3, vR4) ~ (pR2, pR3, pR4)

        # E.g., self.t_2_constrained_pR = {0:0, 1:1, 2:2, 3:3}
        ignored_pR_indices = []
        ignored_vR_indices = []
        ignored_uR_indices = []
        if enforce_t_2_constrained_pR_if_available\
                and self.t_2_constrained_pR is not None:
            ignored_pR_indices = list(self.t_2_constrained_pR.keys())

            ks = self.t_2_constrained_pR.keys()

            # Start from 1 since vR0 is not related to any pR.
            for i in range(1, self.h_safe + 1):
                if i-1 in ks and i in ks:
                    ignored_vR_indices.append(i)

            if 0 in ks and 1 in ks:
                ignored_uR_indices.append(1)
            for i in range(self.h_safe + 1):
                if i in ks and i+1 in ks and i+2 in ks:
                    ignored_uR_indices.append(i+2)
        # assert ignored_pR_indices == [0,1,2,3]
        # assert ignored_vR_indices == [1,2,3]
        # assert ignored_uR_indices == [1,2,3]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Boundary condition for u
        u_min = self.env.uR_min.squeeze().tolist()
        u_max = self.env.uR_max.squeeze().tolist()
        for i in range(self.h_safe - 1):
            # We use i+1 here because ui = us_1_T[i] = uR_{i+1}.
            if i+1 not in ignored_uR_indices:
                # n_uR x 1
                ui = us_1_T[i, :].T
                g = vertcat(g, ui)
                lbg += u_min
                ubg += u_max
                for j in range(len(u_min)):
                    g_name += ["u_bd_i={}_dim={}".format(i, j)]

        # Boundary condition for the terminal u
        # We use self.h_safe here
        # because u_T = us_1_T[self.h_safe - 1] = uR_{self.h_safe}.
        if self.h_safe not in ignored_uR_indices:
            u_T = us_1_T[self.h_safe - 1, :].T
            g = vertcat(g, u_T)
            lbg += u_min
            ubg += u_max
            for j in range(self.env.n_uR):
                g_name += ["u_bd_i={}_dim={}".format(self.h_safe-1, j)]

        # Boundary condition for pR
        pR_min = self.env.pR_min.squeeze().tolist()
        pR_max = self.env.pR_max.squeeze().tolist()
        for i in range(self.h_safe):
            # We use i+1 here because pR = pRs_1_T[i] = pR_{i+1}.
            if i+1 not in ignored_pR_indices:
                # n_pR x 1
                pR = pRs_1_T[i, :].T
                g = vertcat(g, pR)
                lbg += pR_min
                ubg += pR_max
                for j in range(len(pR_min)):
                    g_name += ["pR_bd_i={}_dim={}".format(i, j)]

        # Boundary condition for the terminal pR if we constrain it to the goal
        if terminal_goal_cst:
            # We use self.h_safe here
            # because pR_T = pRs_1_T[self.h_safe - 1] = pR_{self.h_safe}.
            if self.h_safe not in ignored_pR_indices:
                # pR terminal = goal
                pR_T = pRs_1_T[self.h_safe - 1, :].T
                g = vertcat(g, pR_T)
                tmp = np.reshape(np.array([1e-3] * self.env.n_pR),
                                 (self.env.n_pR,))
                lbg += (self.env.pR_goal - tmp).tolist()
                ubg += (self.env.pR_goal + tmp).tolist()
                for j in range(self.env.n_pR):
                    g_name += ["pR_bd_goal_i={}_dim={}".format(self.h_safe-1, j)]

        # Boundary condition for vR
        vR_min = self.env.vR_min.squeeze().tolist()
        vR_max = self.env.vR_max.squeeze().tolist()
        for i in range(self.h_safe - 1):
            # We use i+1 here because vR = vRs_1_T[i] = vR_{i+1}.
            if i+1 not in ignored_vR_indices:
                # n_vR x 1
                vR = vRs_1_T[i, :].T
                g = vertcat(g, vR)
                lbg += vR_min
                ubg += vR_max
                for j in range(len(vR_min)):
                    g_name += ["vR_bd_i={}_dim={}".format(i, j)]

        # Boundary condition for the terminal vR (we set vR terminal = 0)
        # (See the paper Sec.V(B) trajectory optimization formulation (d))
        # We use self.h_safe here
        # because vR_T = vRs_1_T[self.h_safe - 1] = vR_{self.h_safe}.
        if self.h_safe not in ignored_vR_indices:
            vR_T = vRs_1_T[self.h_safe - 1, :].T
            g = vertcat(g, vR_T)
            lbg += [-self.feas_tol] * self.env.n_vR
            ubg += [self.feas_tol] * self.env.n_vR
            for j in range(self.env.n_vR):
                g_name += ["vR_bd_i={}_dim={}".format(self.h_safe-1, j)]

        # We allow user to constrain multiple pRs
        if enforce_t_2_constrained_pR_if_available\
                and self.t_2_constrained_pR is not None:
            for t in self.t_2_constrained_pR.keys():
                # We cannot constrain at t=0,
                # since pR_0 is a parameter, not decision variable.
                if t == 0:
                    continue
                # 2x1
                pR = pRs_1_T[t-1, :].T
                assert pR.shape[1] == 1
                # 2x1
                specified_val = self.t_2_constrained_pR[t].reshape(pR.shape)
                # print("Match pRs_1_T at {} and specified_val at {}"
                      # .format(t-1, t))
                assert pR.shape == specified_val.shape
                tmp = pR - specified_val
                d_sqr = mtimes(tmp.T, tmp)
                assert d_sqr.shape == (1, 1)
                g = vertcat(g, d_sqr)
                lbg += [-1e-3]
                ubg += [1e-3]
                g_name += ["constrained_pR_t={}".format(t)]

        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_robot_dyn_csts(
            self, pRs_1_T, vRs_1_T, us_1_T, pR_0, vR_0, dt,
            ignore_some_csts_if_t_2_constrained_pR_available=True):
        """
        Formulate symbolic constraint for the dynamics with pR,vR,u.

        Parameters
        ----------
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.
        vRs_1_T: horizon x n_pR = robot velocities, vR, along the time steps.
        us_1_T: horizon x n_u = robot controls, u, along the time steps.
        pR_0: initial robot position.
        vR_0: initial robot velocity.
        dt: duration of 1 time step in the robot MPC.
        ignore_some_csts_if_t_2_constrained_pR_available: bool,
            ignore_some_csts_if_t_2_constrained_pR_available = True:
            if self.t_2_constrained_pR is defined,
            then this function will ignore the dynamics constraints for the pR's
            that have been specified by the user in self.t_2_constrained_pR.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, vRs_1_T=vRs_1_T, us_1_T=us_1_T,
                          pR_0=pR_0, vR_0=vR_0)
        g = []
        lbg = []
        ubg = []
        g_name = []

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # For ignore_some_csts_if_t_2_constrained_pR_available:
        # If we have specified self.t_2_constrained_pR,
        # then we can ignore the constraints for the pRs that are
        # constrained by self.t_2_constrained_pR.

        # Suppose h_safe = 4.
        # pR0, [pR1, pR2, pR3, pR4] where [] = pRs_1_T
        # vR0, [vR1, vR2, vR3, vR4] where [] = pRs_1_T
        # vR0 ~ ()
        # vR1 ~ (pR0, pR1)
        # vR2 ~ (pR1, pR2)
        # vR3 ~ (pR2, pR3)
        # vR4 ~ (pR3, pR4)
        # [uR1, uR2, uR3, uR4] where [] = us_1_T
        # uR1 ~ (vR0, vR1) ~ (pR0, pR1)
        # uR2 ~ (vR1, vR2) ~ (pR0, pR1, pR2)
        # uR3 ~ (vR2, vR3) ~ (pR1, pR2, pR3)
        # uR4 ~ (vR3, vR4) ~ (pR2, pR3, pR4)

        # E.g., self.t_2_constrained_pR = {0:0, 1:1, 2:2, 3:3}
        ignored_vR_indices = []
        ignored_uR_indices = []
        if ignore_some_csts_if_t_2_constrained_pR_available\
                and self.t_2_constrained_pR is not None:
            ks = self.t_2_constrained_pR.keys()

            # Start from 1 since vR0 is not related to any pR.
            for i in range(1, self.h_safe + 1):
                if i-1 in ks and i in ks:
                    ignored_vR_indices.append(i)

            if 0 in ks and 1 in ks:
                ignored_uR_indices.append(1)
            for i in range(self.h_safe + 1):
                if i in ks and i+1 in ks and i+2 in ks:
                    ignored_uR_indices.append(i+2)
        # assert ignored_vR_indices == [1,2,3]
        # assert ignored_uR_indices == [1,2,3]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Dynamics: pR ~ vR
        # We use 1 here because next_vR = vRs_1_T[0] = vR1.
        if 1 not in ignored_vR_indices:
            # npRx1
            cur_pR = pR_0
            next_pR = pRs_1_T[0, :].T
            next_vR = vRs_1_T[0, :].T
            assert cur_pR.shape == next_pR.shape == next_vR.shape
            tmp = next_pR - cur_pR - dt * next_vR
            # print(" ".join([str(i.shape) for i in [next_pR,
            # cur_pR, next_vR, tmp, vR_0]]))
            assert next_pR.shape == tmp.shape == vR_0.shape
            g = vertcat(g, tmp)
            lbg += [-self.feas_tol] * self.env.n_pR
            ubg += [self.feas_tol] * self.env.n_pR
            for j in range(self.env.n_pR):
                g_name += ["pR_dyn_t-1_vs_t0_dim={}".format(j)]

        for i in range(self.h_safe - 1):
            # We use i+2 because next_vR = vRs_1_T[i+1] = vR_{i+2}.
            if i+2 not in ignored_vR_indices:
                cur_pR = pRs_1_T[i, :].T
                next_pR = pRs_1_T[i+1, :].T
                next_vR = vRs_1_T[i+1, :].T
                tmp = next_pR - cur_pR - dt * next_vR
                assert next_pR.shape == cur_pR.shape\
                    == next_vR.shape == tmp.shape
                g = vertcat(g, tmp)
                lbg += [-self.feas_tol] * self.env.n_pR
                ubg += [self.feas_tol] * self.env.n_pR
                for j in range(self.env.n_pR):
                    g_name += ["pR_dyn_t{}_vs_t{}_dim={}".format(i, i+1, j)]

        # Dynamics: vR ~ u
        # We use 1 here because next_u = us_1_T[0] = uR1.
        if 1 not in ignored_uR_indices:
            cur_vR = vR_0
            next_vR = vRs_1_T[0, :].T
            next_u = us_1_T[0, :].T
            tmp = next_vR - cur_vR - dt * next_u / self.env.mR
            assert next_u.shape == next_vR.shape == cur_vR.shape == tmp.shape
            g = vertcat(g, tmp)
            lbg += [-self.feas_tol] * self.env.n_vR
            ubg += [self.feas_tol] * self.env.n_vR
            for j in range(self.env.n_vR):
                g_name += ["vR_dyn_t-1_vs_t0_dim={}".format(j)]

        for i in range(self.h_safe - 1):
            # We use i+2 because next_u = us_1_T[i+1] = uR_{i+2}.
            if i+2 not in ignored_uR_indices:
                cur_vR = vRs_1_T[i, :].T
                next_vR = vRs_1_T[i+1, :].T
                next_u = us_1_T[i+1, :].T
                tmp = next_vR - cur_vR - dt * next_u / self.env.mR
                assert next_u.shape == next_vR.shape\
                    == cur_vR.shape == tmp.shape
                g = vertcat(g, tmp)
                lbg += [-self.feas_tol] * self.env.n_vR
                ubg += [self.feas_tol] * self.env.n_vR
                for j in range(self.env.n_vR):
                    g_name += ["vR_dyn_t{}_vs_t{}_dim={}".format(i, i+1, j)]

        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_control_cost(self, us_1_T):
        """
        Formulate symbolic cost function - to penalize total control.

        Parameters
        ----------
        us_1_T: horizon x n_u = robot controls, u, along the time steps.

        Returns
        ----------
        cost: symbolic cost variable.
        """

        self.check_shapes(us_1_T=us_1_T)

        cost = 0
        for i in range(self.h_safe):
            ui = us_1_T[i, :].T
            assert ui.shape == (self.env.n_uR, 1)
            cost += mtimes(ui.T, mtimes(self.w_u, ui))
        assert cost.shape == (1, 1)
        return cost

    def generate_terminal_goal_cost(self, pRs_1_T, pR_goal=None):
        """
        Formulate symbolic cost function
        - to penalize the distance to the goal at the end of the trajectory.

        Parameters
        ----------
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.
        pR_goal: n_pR x 1 = robot goal position.

        Returns
        ----------
        cost: symbolic cost variable.
        """

        self.check_shapes(pRs_1_T=pRs_1_T)

        if pR_goal is None:
            goal = np.reshape(self.env.pR_goal, (self.env.n_pR, 1))
        else:
            goal = pR_goal
        pR_T = pRs_1_T[self.h_safe - 1, :].T
        assert goal.shape == pR_T.shape == (self.env.n_pR, 1)
        cost = mtimes((pR_T - goal).T, mtimes(self.w_goal, (pR_T - goal)))
        assert cost.shape == (1, 1)
        return cost

    def generate_d2arm_cost(self, pRs_1_T, pHs_1_T):
        """
        Formulate symbolic cost function (only for the 2D dressing task).
        - to penalize the distance from the robot, pR, to the human arm,
          where the human arm is approximated as a line segment
          between pH and the fixed human shouolder position.
          This cost is motivated by the armhole with a limited size.
          Hence, the robot cannot stay very far from the human arm.

        Parameters
        ----------
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.

        Returns
        ----------
        cost: symbolic cost variable.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T)
        cost = 0
        for i in range(self.h_safe):
            # 2x1
            pR = pRs_1_T[i, :].T
            # 2x1
            pH = pHs_1_T[i, :].T
            pH_sd = np.reshape(self.env.pH_shoulder, pH.shape)
            x0 = pR
            x1 = pH
            x2 = pH_sd
            d_sqr = dsqr_x0_2_line_x1x2(x0=x0, x1=x1, x2=x2)
            assert d_sqr.shape == (1, 1)
            cost += self.w_dist_2_line_pH_pH_shoulder * d_sqr
        assert cost.shape == (1, 1)
        return cost

    def generate_approach_pH_cost(self, pRs_1_T, pHs_1_T):
        """
        Formulate symbolic cost function (only for the handover task).
        - to motivate the robot to approach pH at each time step.

        Parameters
        ----------
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.

        Returns
        ----------
        cost: symbolic cost variable.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T)
        cost = 0
        for i in range(self.h_safe):
            pR = pRs_1_T[i, :].T
            pH = pHs_1_T[i, :].T
            pH = cas_reshape(pH, pR.shape)
            assert pR.shape == pH.shape
            assert pR.shape[1] == 1
            tmp = pR - pH
            d_sqr = mtimes(tmp.T, tmp)
            assert d_sqr.shape == (1, 1)
            cost += self.w_dist_2_pH * d_sqr
            # print("cur={}".format(i))
        assert cost.shape == (1, 1)
        return cost

    def get_qp_traj(self, pR_0, vR_0, dt, horizon, plot=False):
        """
        Solve Quadratic Programming (QP) trajectory optimization problem.
        The QP has simplified constraints, compared with the safe MPC.
        The solution of the QP will be used as the initial guess for the safe MPC.

        Parameters
        ----------
        pR_0: initial robot position.
        vR_0: initial robot velocity.
        dt: duration of 1 time step in the robot MPC.
        horizon: length of the trajectory

        Returns
        ----------
        pRs_1_T_opt: horizon x n_pR = optimal trajectory of the robot position, pR.
        vRs_1_T_opt: horizon x n_pR = optimal trajectory of the robot velocity, vR.
        us_1_T_opt: horizon x n_pR = optimal trajectory of the robot control, u.
        """

        self.check_shapes(pR_0=pR_0, vR_0=vR_0)

        # Decision variables
        # T = self.h_safe
        pRs_1_T = MX.sym("pRs_1_T", (horizon, self.env.n_pR))
        vRs_1_T = MX.sym("vRs_1_T", (horizon, self.env.n_vR))
        us_1_T = MX.sym("us_1_T", (horizon, self.env.n_uR))

        # Constraints
        g = []
        lbg = []
        ubg = []
        g_name = []

        g_bd, lbg_bd, ubg_bd, g_names_bd = self.generate_boundary_csts_bounds(
                pRs_1_T, vRs_1_T, us_1_T, terminal_goal_cst=False,
                enforce_t_2_constrained_pR_if_available=False)
        g = vertcat(g, g_bd)
        lbg += lbg_bd
        ubg += ubg_bd
        g_name += g_names_bd

        g_dyn, lbg_dyn, ubg_dyn, g_names_dyn = self.generate_robot_dyn_csts(
                pRs_1_T, vRs_1_T, us_1_T, pR_0, vR_0, dt)
        g = vertcat(g, g_dyn)
        lbg += lbg_dyn
        ubg += ubg_dyn
        g_name += g_names_dyn

        cost = self.generate_control_cost(us_1_T)
        cost += self.generate_terminal_goal_cost(pRs_1_T)
        assert cost.shape == (1, 1)

        # All variables and parameters
        opt_vars = vertcat(
                pRs_1_T.reshape((-1, 1)),
                vRs_1_T.reshape((-1, 1)),
                us_1_T.reshape((-1, 1)))
        prob = {'f': cost, 'x': opt_vars, 'g': g}
        # https://github.com/casadi/casadi/blob/master/docs/examples/python/chain_qp.py
        opt = {'error_on_fail': False,
               'sparse': True,
               'printLevel': "none"}
        solver = cas.qpsol('solver', 'qpoases', prob, opt)

        # Straight line path as a guess
        end_time = dt * horizon
        pRs_ref = np.vstack((pR_0.T, self.env.pR_goal.T))
        assert pRs_ref.shape == (2, self.env.n_pR)
        pw_spliness, dts_pw = waypts_2_pwsplines(
                wp_traj=pRs_ref, dt=end_time,
                degree=1, plot=False)
        dts = np.linspace(0, end_time, num=horizon+1, endpoint=True)
        # (horizon + 1) x npR
        pRs_0_T_init = np.zeros((dts.shape[0], self.env.n_pR))
        for i in range(self.env.n_pR):
            tmp = interpolate.splev(x=dts, tck=pw_spliness[i], ext=2)
            assert pRs_0_T_init[:, i].shape == tmp.shape
            pRs_0_T_init[:, i] = tmp
        assert (pRs_0_T_init[0, :] == pR_0.T).all()
        pRs_1_T_init = pRs_0_T_init[1:, :]
        assert pRs_1_T_init.shape == pRs_1_T.shape

        pRs_0_T_minus_1_init = pRs_0_T_init[:-1, :]
        us_1_T_init = (pRs_1_T_init - pRs_0_T_minus_1_init) / dt
        assert us_1_T_init.shape == us_1_T.shape

        vRs_1_T_init = np.zeros((horizon, self.env.n_vR))
        assert vRs_1_T_init.shape == vRs_1_T.shape

        opt_vars_init = vertcat(
                pRs_1_T_init.reshape((-1, 1)),
                vRs_1_T_init.reshape((-1, 1)),
                us_1_T_init.reshape((-1, 1)))
        assert opt_vars.shape == opt_vars_init.shape

        sol = solver(x0=opt_vars_init, lbg=lbg, ubg=ubg)
        x_opt = sol['x']
        f_opt = sol['f']
        print('f_opt = ', f_opt)

        g_res = np.array(sol["g"]).squeeze()
        feasible = True
        # This is not sufficient,
        # since casadi gives out wrong feasibility values
        if np.any(np.array(lbg) - self.feas_tol > g_res) or np.any(
                np.array(ubg) + self.feas_tol < g_res):
            feasible = False

        if not feasible:
            # Return init straight line path
            print_FAIL("QP is not feasible, check u / vel bound")
            return pRs_1_T_init, vRs_1_T_init, us_1_T_init

        # get indices of the respective variables
        c = 0
        n_pRs_1_T = horizon * self.env.n_pR
        idx_pRs_1_T = np.arange(n_pRs_1_T)
        c += n_pRs_1_T
        n_vRs_1_T = horizon * self.env.n_vR
        idx_vRs_1_T = np.arange(c, c + n_vRs_1_T)
        c += n_vRs_1_T
        n_us_1_T = horizon * self.env.n_uR
        idx_us_1_T = np.arange(c, c + n_us_1_T)
        c += n_us_1_T
        assert c == x_opt.shape[0]

        pRs_1_T_opt = np.array(cas_reshape(x_opt[idx_pRs_1_T],
                               (horizon, self.env.n_pR)))
        assert pRs_1_T.shape == pRs_1_T_opt.shape
        vRs_1_T_opt = np.array(cas_reshape(x_opt[idx_vRs_1_T],
                               (horizon, self.env.n_vR)))
        assert vRs_1_T.shape == vRs_1_T_opt.shape
        us_1_T_opt = np.array(cas_reshape(x_opt[idx_us_1_T],
                              (horizon, self.env.n_uR)))
        assert us_1_T.shape == us_1_T_opt.shape

        if plot:
            assert self.env.n_pR == self.env.n_vR == self.env.n_uR
            for i in range(self.env.n_pR):
                pRs_0_T_opt = np.vstack(
                        (np.reshape(pR_0, (self.env.n_pR)), pRs_1_T_opt))
                vRs_0_T_opt = np.vstack(
                        (np.reshape(vR_0, (self.env.n_vR)), vRs_1_T_opt))
                us_0_T_opt = np.vstack((np.zeros(self.env.n_vR), us_1_T_opt))
                # plt.figure()
                fig, axs = plt.subplots(3)
                fig.tight_layout()
                axs[0].plot(dts, pRs_0_T_opt[:, i], 'ro')
                axs[0].plot(dts, pRs_0_T_opt[:, i], 'g')
                axs[0].set_title('pRs_0_T_opt')
                axs[1].plot(dts, vRs_0_T_opt[:, i], 'ro')
                axs[1].plot(dts, vRs_0_T_opt[:, i], 'g')
                axs[1].set_title('vRs_0_T_opt')
                axs[2].plot(dts, us_0_T_opt[:, i], 'ro')
                axs[2].plot(dts, us_0_T_opt[:, i], 'g')
                axs[2].set_title('us_0_T_opt')
                tmp = 'QP_pvu_dim_{}.pdf'.format(i)
                path = os.path.join(self.result_dir, tmp)
                plt.savefig(path)
                print("Savefig to {}".format(path))
                plt.cla()
                plt.clf()
                plt.close()

        return pRs_1_T_opt, vRs_1_T_opt, us_1_T_opt

    def solve_MPC(self, pR_0, vR_0, pH_0, itr_idx):
        """
        Solve the safe MPC trajectory optimization problem.
        - If the MPC is feasible, return the found trajectory.
        - If the MPC is infeasible, then
          - Shift the trajectory from the previous iteration
            and return the shifted previous trajectory.
          - If the previous trajectory is too short to be shifted,
            then return the recovery controller.

        Parameters
        ----------
        pR_0: initial robot position.
        vR_0: initial robot velocity.
        pH_0: initial human position.
        itr_idx: the index of the current iteration in the iterative MPC.
                 (used for logging purposes).

        Returns
        (all the followings are computed as part of the MPC's solution).
        ----------
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.
        qHs_1_T: horizon x n_pH^2 = shape of ellipsoids for pH along the time steps.
        sH_gp_pred_sigma_1_T: (T x n_pH) x 1
                              = the GP predictive stddev of pH along the time steps.
        vHs_1_T: horizon x n_pH = centers of ellipsoids for vH along the time steps.
        vqHs_1_T: horizon x n_pH^2 = shape of ellipsoids for vH along the time steps.

        pRs_1_T_opt: horizon x n_pR = optimal trajectory of the robot position, pR.
        vRs_1_T_opt: horizon x n_pR = optimal trajectory of the robot velocity, vR.
        us_1_T_opt: horizon x n_pR = optimal trajectory of the robot control, u.
        status: str, status of the MPC solver,
                indicating whether a new trajectory is returned,
                or an old shifted trajectory is returned,
                or the recovery controller is returned.
        planning_time: float, planning time (sec).
        """

        self.check_shapes(pR_0=pR_0, vR_0=vR_0, pH_0=pH_0)
        self.itr_idx = itr_idx

        # We don't optimize dt, beta. So they are parameters.
        dt = self.dt
        beta = self.fixed_beta

        pRs_1_T_init, vRs_1_T_init, us_1_T_init\
            = self._get_init_controls(pR_0, vR_0, dt)

        opt_vars_init = None
        params = None
        opt_vars_init = vertcat(
                pRs_1_T_init.reshape((-1, 1)),
                vRs_1_T_init.reshape((-1, 1)),
                us_1_T_init.reshape((-1, 1)))
        params = np.vstack((pR_0, vR_0, pH_0, beta, dt))
        assert self.prob['p'].shape == params.shape
        assert self.prob['x'].shape == opt_vars_init.shape

        start = time.time()
        sol = self.solver(x0=opt_vars_init,
                          lbg=self.lbg, ubg=self.ubg, p=params)
        end = time.time()
        planning_time = end - start
        crash = False

        feasible = True
        if crash:
            feasible = False
            print("Optimization crashed, infeasible soluion!")
        else:
            g_res = np.array(sol["g"]).squeeze()

            # This is not sufficient,
            # since casadi gives out wrong feasibility values
            if np.any(np.array(self.lbg) - self.feas_tol > g_res) or np.any(
                    np.array(self.ubg) + self.feas_tol < g_res):
                lb_vios = np.argwhere(
                        g_res < np.array(self.lbg) - self.feas_tol)
                if lb_vios.shape[0] == 1:
                    lb_vio_names = [self.g_name[np.squeeze(lb_vios)]]
                else:
                    lb_vio_names = [self.g_name[i] for i in np.squeeze(lb_vios)]

                ub_vios = np.argwhere(
                        g_res > np.array(self.ubg) + self.feas_tol)
                if ub_vios.shape[0] == 1:
                    ub_vio_names = [self.g_name[np.squeeze(ub_vios)]]
                else:
                    ub_vio_names = [self.g_name[i] for i in np.squeeze(ub_vios)]
                print_FAIL("Infeasible <= CAS cst violation")
                print("lb_vio_names={}".format(lb_vio_names))
                print("ub_vio_names={}".format(ub_vio_names))
                feasible = False
            else:
                print_OK("Feasible <= CAS cst violation")

            x_opt = sol["x"]

            # get indices of the respective variables
            c = 0
            n_pRs_1_T = self.h_safe * self.env.n_pR
            idx_pRs_1_T = np.arange(n_pRs_1_T)
            c += n_pRs_1_T
            n_vRs_1_T = self.h_safe * self.env.n_vR
            idx_vRs_1_T = np.arange(c, c + n_vRs_1_T)
            c += n_vRs_1_T
            n_us_1_T = self.h_safe * self.env.n_uR
            idx_us_1_T = np.arange(c, c + n_us_1_T)
            c += n_us_1_T

            assert c == x_opt.shape[0]

            pRs_1_T_opt = np.array(cas_reshape(x_opt[idx_pRs_1_T],
                                   (self.h_safe, self.env.n_pR)))
            assert pRs_1_T_init.shape == pRs_1_T_opt.shape
            vRs_1_T_opt = np.array(cas_reshape(x_opt[idx_vRs_1_T],
                                   (self.h_safe, self.env.n_vR)))
            assert vRs_1_T_init.shape == vRs_1_T_opt.shape
            us_1_T_opt = np.array(cas_reshape(x_opt[idx_us_1_T],
                                  (self.h_safe, self.env.n_uR)))
            assert us_1_T_init.shape == us_1_T_opt.shape

            pHs_1_T = None
            qHs_1_T = None
            sH_gp_pred_sigma_1_T = None
            vHs_1_T = None
            vqHs_1_T = None
            if self.pR_mode == "CA":
                pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T\
                    = self._f_multistep_eval(pH_0, pR_0, pRs_1_T_opt, beta)
            elif self.pR_mode == "CA_SI":
                pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T,\
                    vHs_1_T, vqHs_1_T = self._f_multistep_vel_eval(
                            pH_0, pR_0, pRs_1_T_opt, beta, dt)

            if feasible:
                # Save trajectory for later when we are doing receding horizon.
                # e.g., use this trajectory again in the next itr,
                # if MPC is not feasible then.
                self.pRs_1_T_opt = pRs_1_T_opt
                self.vRs_1_T_opt = vRs_1_T_opt
                self.us_1_T_opt = us_1_T_opt
            else:
                print_FAIL("*** solve_MPC is Infeasible!")

        if feasible:
            self.n_fail = 0
            status = self.status_feasible_MPC

            # Return the optimal sol
            return pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T,\
                vHs_1_T, vqHs_1_T,\
                pRs_1_T_opt, vRs_1_T_opt, us_1_T_opt,\
                status, planning_time
        else:
            self.n_fail += 1
            if self.n_fail >= self.h_safe:
                # Too many infeasible solutions -> switch to recovery controller
                print(("Infeasible solution. Too many. Use recovery controller, "
                       + "n_fail={}, h_safe={}"
                      .format(self.n_fail, self.h_safe)))
                status = self.status_infeasible_MPC_use_safe_policy

                # Return the infeasible sol
                return pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T,\
                    vHs_1_T, vqHs_1_T,\
                    pRs_1_T_opt, vRs_1_T_opt, us_1_T_opt,\
                    status, planning_time
            else:
                # can apply previous solution
                print(("Infeasible solution. Use previous sol, "
                       + "n_fail={}, h_safe={}"
                      .format(self.n_fail, self.h_safe)))
                print_FAIL("Note that if we update GP at every round, "
                           + "we might not be able to use the prev solution")

                pRs_1_T, vRs_1_T, us_1_T = self.shift_sol_forward_1_step(
                        pR_0, vR_0, self.us_1_T_opt, self.dt,
                        num_shift=self.shift_num)

                pHs_1_T = None
                qHs_1_T = None
                sH_gp_pred_sigma_1_T = None
                vHs_1_T = None
                vqHs_1_T = None
                if self.pR_mode == "CA":
                    pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T\
                        = self._f_multistep_eval(
                            pH_0, pR_0, pRs_1_T, self.fixed_beta)
                elif self.pR_mode == "CA_SI":
                    pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T,\
                        vHs_1_T, vqHs_1_T = self._f_multistep_vel_eval(
                            pH_0, pR_0, pRs_1_T, self.fixed_beta, self.dt)

                status = self.status_infeasible_MPC_use_old_traj

                self.pRs_1_T_opt = pRs_1_T
                self.vRs_1_T_opt = vRs_1_T
                self.us_1_T_opt = us_1_T

                # Return the shifted previous sol
                return pHs_1_T, qHs_1_T, sH_gp_pred_sigma_1_T,\
                    vHs_1_T, vqHs_1_T,\
                    self.pRs_1_T_opt, self.vRs_1_T_opt, self.us_1_T_opt,\
                    status, planning_time

    def eval_coll_avoid(self, pRs_1_T, pHs_1_T, qHs_1_T):
        """
        Evaluate collision avoidance constraints on the found trajectory.

        Parameters
        ----------
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.
        qHs_1_T: horizon x n_pH^2 = shape of ellipsoids for pH along the time steps.

        Returns
        ----------
        feasible_coll_avoid: bool, whether the trajectory is feasible or not.
        invalid_times_coll_avoid: list, the times when the state is infeasible.
        dists: list, the human-robot velocity distance of the trajectory across all time steps.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T, qHs_1_T=qHs_1_T)
        n_pts = self.env.R_col_volume_offsets.shape[0]

        dists = np.zeros((self.h_safe, n_pts))
        for t in range(self.h_safe):
            # 2x1
            pR = np.reshape(pRs_1_T[t, :].T, (self.env.n_pR, 1))
            # 2x1
            pH = np.reshape(pHs_1_T[t, :].T, (self.env.n_pH, 1))
            # 4x1
            qH = qHs_1_T[t, :].T

            Jd = self.compute_Jac_4_pR_pH_qH_dist_cas_fcn(pR, pH, qH)
            Jd_pR = Jd[0].toarray()
            Jd_pH = Jd[1].toarray()
            Jd_qH = Jd[2].toarray()
            assert not (np.isnan(Jd_pR)).any()
            assert not (np.isnan(Jd_pH)).any()
            assert not (np.isnan(Jd_qH)).any()

            d = self.compute_pR_pH_qH_dist_cas_fcn(pR, pH, qH)
            assert d.shape == (n_pts, 1)
            d = d.toarray()
            d = d.reshape((n_pts,))
            assert d.shape == dists[t, :].shape
            dists[t, :] = d

        dists = dists.squeeze()
        # Safe: dist >= 1
        invalid_times_coll_avoid = np.where(dists < 1.)[0].tolist()
        for t in range(self.h_safe):
            dist = dists[t]
            if dist >= 1.:
                assert t not in invalid_times_coll_avoid
            else:
                assert t in invalid_times_coll_avoid
        feasible_coll_avoid = len(invalid_times_coll_avoid) <= 0
        return feasible_coll_avoid, invalid_times_coll_avoid, dists

    def eval_safe_impact(self, vRs_1_T, vHs_1_T, vqHs_1_T):
        """
        Evaluate safe impact constraints on the found trajectory.

        Parameters
        ----------
        vRs_1_T: horizon x n_pR = robot velocities, vR, along the time steps.
        vHs_1_T: horizon x n_pH = centers of ellipsoids for vH along the time steps.
        vqHs_1_T: horizon x n_pH^2 = shape of ellipsoids for vH along the time steps.

        Returns
        ----------
        feasible_safe_impact: bool, whether the trajectory is feasible or not.
        invalid_times_safe_impact: list, the times when the state is infeasible.
        ds: list, the human-robot position distance of the trajectory across all time steps.
        """

        self.check_shapes(vRs_1_T=vRs_1_T, pHs_1_T=vHs_1_T,
                          qHs_1_T=vqHs_1_T)
        # (2*nvR)xnvR = 4x2
        h_mat = self.env.h_mat_safe_imp_pot
        n_cst = h_mat.shape[0]
        assert h_mat.shape == (n_cst, self.env.n_vR)

        invalid_times_safe_impact = []
        valid_times_safe_impact = []
        ds = np.zeros((self.h_safe, n_cst))
        for t in range(self.h_safe):
            # 2x1
            vR = np.reshape(vRs_1_T[t, :], (self.env.n_vR, 1))
            # 2x1
            vH = vHs_1_T[t, :].T
            # 4x1
            vqH = vqHs_1_T[t, :].T
            vqH = np.reshape(vqH, (self.env.n_pH, self.env.n_pH))

            # Ensure [Hx]i * x <= hix.
            d = self.compute_lin_ellipsoid_safety_distance_cas_fcn(vR, vH, vqH)
            d = d.toarray()
            assert d.shape == (n_cst, 1)

            d = d.reshape((n_cst,))
            assert d.shape == ds[t, :].shape
            ds[t, :] = d

            # We want d = Hi*p + \sqrt(Hi*Qt*Hi) - hi \leq 0
            if (d <= 0.).all():
                valid_times_safe_impact.append(t)
                # print("Safe d safe impact={}".format(d))
            else:
                # print("Unsafe d safe impact={}".format(d))
                invalid_times_safe_impact.append(t)
        feasible_safe_impact = len(invalid_times_safe_impact) <= 0
        return feasible_safe_impact, invalid_times_safe_impact, ds

    def eval_d2arm_csts(self, pHs_1_T, pRs_1_T):
        """
        Evaluate <distance to human arm> constraints on the found trajectory.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.
        pRs_1_T: horizon x n_pR = robot positions, pR, along the time steps.

        Returns
        ----------
        feasible: bool, whether the trajectory is feasible or not.
        invalid_times: list, the times when the state is infeasible.
        ds: list, the distance between the robot and human arm across all time steps.
        """

        self.check_shapes(pRs_1_T=pRs_1_T, pHs_1_T=pHs_1_T)
        invalid_times = []
        valid_times = []
        ds = []
        for t in range(self.h_safe):
            # 2x1
            pR = pRs_1_T[t, :].T
            # 2x1
            pH = pHs_1_T[t, :].T
            pH_sd = np.reshape(self.env.pH_shoulder, pH.shape)
            x0 = pR
            x1 = pH
            x2 = pH_sd
            d_sqr = self.compute_dsqr_x0_2_line_x1x2_cas_fcn(x0, x1, x2)
            assert d_sqr.shape == (1, 1)
            if d_sqr <= self.env.max_dist_bw_pR_arm**2:
                valid_times.append(t)
            else:
                invalid_times.append(t)
            ds.append(np.sqrt(d_sqr))
            # print("pR={}, pH={},{}, d={}".format(
                # x0, x1, x2, np.sqrt(d_sqr)))
        feasible = len(invalid_times) <= 0
        return feasible, invalid_times, ds

    def _get_init_controls(self, pR_0, vR_0, dt):
        """
        Use Quadratic Programming (QP) with simplified constraints
        to initialize the robot trajectory and control for the safe MPC.

        Parameters
        ----------
        pR_0: initial robot position.
        vR_0: initial robot velocity.
        dt: duration of 1 time step in the robot MPC.

        Returns
        ----------
        pRs_1_T: horizon x n_pR = QP solution trajectory of the robot position, pR.
        vRs_1_T: horizon x n_pR = QP solution trajectory of the robot velocity, vR.
        us_1_T: horizon x n_pR = QP solution trajectory of the robot control, u.
        """

        self.check_shapes(pR_0=pR_0, vR_0=vR_0)
        pRs_1_T, vRs_1_T, us_1_T = self.get_qp_traj(
                pR_0, vR_0, dt, horizon=self.h_safe, plot=False)
        self.check_shapes(pRs_1_T=pRs_1_T, vRs_1_T=vRs_1_T,
                          us_1_T=us_1_T, pR_0=pR_0, vR_0=vR_0)
        return pRs_1_T, vRs_1_T, us_1_T

    def shift_sol_forward_1_step(self, pR_0, vR_0, us_1_T, dt, num_shift=1):
        """
        Shift solution trajectory forward for num_shift step.

        Parameters
        ----------
        pR_0: initial robot position.
        vR_0: initial robot velocity.
        us_1_T: horizon x n_u = robot controls, u, along the time steps.
        dt: duration of 1 time step in the robot MPC.
        num_shift: number of steps to shift.

        Returns
        ----------
        pRs_1_T: horizon x n_pR = shifted trajectory of the robot position, pR.
        vRs_1_T: horizon x n_pR = shifted trajectory of the robot velocity, vR.
        us_1_T: horizon x n_pR = shifted trajectory of the robot control, u.
        """

        self.check_shapes(pR_0=pR_0, vR_0=vR_0, us_1_T=us_1_T)

        assert us_1_T.shape[0] > num_shift
        # Shift by num_shift and pad with safe policy
        us_1_T = np.copy(self.us_1_T_opt)
        us_1_T = np.roll(us_1_T, shift=-num_shift, axis=0)
        us_1_T[-num_shift:, :] = self.safe_policy(pR_0, vR_0, None, None)

        # Integrate forward to get the trajectory
        pRs_1_T, vRs_1_T = self.rollout_pR_traj(
                pR_0, vR_0, us_1_T, dt)
        return pRs_1_T, vRs_1_T, us_1_T

    def rollout_pR_traj(self, pR_0, vR_0, us_1_T, dt):
        """
        Rollout out us_1_T throughout the time steps to find the pR,vR trajectory.

        Parameters
        ----------
        pR_0: initial robot position.
        vR_0: initial robot velocity.
        us_1_T: horizon x n_u = robot controls, u, along the time steps.
        dt: duration of 1 time step in the robot MPC.

        Returns
        ----------
        pRs_1_T: horizon x n_pR = rollout trajectory of the robot position, pR.
        vRs_1_T: horizon x n_pR = rollout trajectory of the robot velocity, vR.
        """

        self.check_shapes(pR_0=pR_0, vR_0=vR_0, us_1_T=us_1_T)

        horizon = us_1_T.shape[0]

        pRs_1_T = np.zeros((horizon, self.env.n_pR))
        vRs_1_T = np.zeros((horizon, self.env.n_vR))
        cur_pR = pR_0
        cur_vR = vR_0
        for i in range(horizon):
            next_u = np.reshape(us_1_T[i, :], (self.env.n_uR, 1))
            next_vR = cur_vR + dt * next_u / self.env.mR
            next_pR = cur_pR + dt * next_vR
            pRs_1_T[i, :] = np.reshape(next_pR, (self.env.n_pR,))
            vRs_1_T[i, :] = np.reshape(next_vR, (self.env.n_vR,))
            cur_pR = next_pR
            cur_vR = next_vR
        return pRs_1_T, vRs_1_T

    def interp_ellipsoids(self, pHs_1_T):
        """
        In dressing 2D, find the centers of a sequence of ellipsoids
        to approximate the human arm.
        The centers are computed via interpolating the line segment
        between the human hand and human shoulder.
        Later, it duplicates the ellipsoid for the human hand
        along the line segment between the human hand and human shoulder.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH = centers of ellipsoids for pH along the time steps.

        Returns
        ----------
        interp_2_pHs_1_T: num_interpolated_points x horizon x n_pH
                          = the interpolated centers.
        """

        self.check_shapes(pHs_1_T=pHs_1_T)
        use_np_array = False
        if type(pHs_1_T) is np.ndarray:
            use_np_array = True

        # Not include pHs_1_T
        interp_2_pHs_1_T = {}

        dts = np.linspace(start=0., stop=1.,
                          num=self.env.n_interp_pH_pH_shoulder + 2,
                          endpoint=True)
        dts = dts[1:-1]

        for dt in dts:
            # At every time, linear interpolate pH, and pH_shoulder
            pHs_1_T_interp = None
            if not use_np_array:
                pHs_1_T_interp = MX.zeros(pHs_1_T.shape)
            else:
                pHs_1_T_interp = np.zeros(pHs_1_T.shape)
            for t in range(self.h_safe):
                # 2x1
                pH = pHs_1_T[t, :].T
                if type(pH) is np.ndarray:
                    pH = np.reshape(pH, (self.env.n_pH, 1))
                assert pH.shape == (self.env.n_pH, 1)\
                    == self.env.pH_shoulder.shape
                pH_interp = pH*(1-dt) + self.env.pH_shoulder*dt
                if not use_np_array:
                    assert pHs_1_T_interp[t, :].shape == pH_interp.T.shape
                pHs_1_T_interp[t, :] = pH_interp.T
            interp_2_pHs_1_T[dt] = pHs_1_T_interp
        return interp_2_pHs_1_T

    def update_model(self, x, y, opt_hyp, replace_old, reinitialize_solver):
        """
        Update GP model.

        Parameters
        ----------
        x: n x (n_s+n_u) array[float]
            The raw training input (state,action) pairs
        y: n x (n_s) array[float]
            The raw training targets (observations of next state)
        opt_hyp: bool
            True, if we want to re-optimize the GP hyperparameters
        replace_old: bool
            True, if we replace the current training set of the GP with x,y
        reinitialize_solver:
            True, if we re-initialize the solver (otherwise the MPC will not be updated with the new GP)
        """
        # self.n_s = 2, self.n_u = 1, x = 25 x 3, y = 25 x 2
        # 25
        n_train = np.shape(x)[0]

        # Here we optionally apply linear transformation
        # based on lin_trafo_gp_input to the data inputs
        # (x's) and recombine the data to be x.
        x_s = x[:, :self.ssm.n_s_in].reshape((n_train, self.ssm.n_s_in))
        x_u = x[:, self.ssm.n_s_in:].reshape((n_train, self.ssm.n_u))
        # self.lin_trafo_gp_input = I, so x_trafo=x.
        x_trafo = mtimes(x_s, self.lin_trafo_gp_input.T)
        x = np.hstack((x_trafo, x_u))

        # Train with data
        self.ssm.update_model(x, y, opt_hyp, replace_old)
        self.ssm_forward = self.ssm.get_forward_model_casadi(True)

        if reinitialize_solver:
            self.init_solver()
        else:
            print_FAIL("Updating gp without reinitializing the solver! This is potentially dangerous, since the new GP is not incorporated in the MPC")
