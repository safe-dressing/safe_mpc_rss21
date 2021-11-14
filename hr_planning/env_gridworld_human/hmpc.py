# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''
import casadi as cas
import numpy as np
from casadi import MX, mtimes, vertcat, sum2, sqrt, jacobian
from casadi import reshape as cas_reshape
from scipy import interpolate
import matplotlib.pyplot as plt

import hr_planning
from hr_planning.visualization.utils_visualization import print_FAIL
from hr_planning.utils_interp import waypts_2_pwsplines

use_human_ui_bound = True
# use_human_ui_bound = False


class HumanRefTracker:
    def __init__(self, pH_view_pH_pR_min_sep_dist,
                 pH_min, pH_max, vH_min, vH_max, uH_min, uH_max,
                 w_ref, w_u, w_move_to_pR, w_avoid_pR, dt, mass):
        """
        Class for an MPC to compute human trajectory in continuous state space.
        n_pH = dimensionality of human position.
        n_vH = dimensionality of human velocity.
        n_uH = dimensionality of human control.

        Parameters
        ----------
        pH_view_pH_pR_min_sep_dist: float, from the human's view,
                                    the min separation distance that the human
                                    wants to keep from the robot.
        The boundary conditions:
        pH_max: (n_pH,) np vector = max human position.
        pH_min: (n_pH,) np vector = min human position.
        vH_max: (n_pH,) np vector = max human velocity.
        vH_min: (n_pH,) np vector = min human velocity.
        uH_max: (n_pH,) np vector = max human control.
        uH_min: (n_pH,) np vector = min human control.
        """

        self.pH_view_pH_pR_min_sep_dist = pH_view_pH_pR_min_sep_dist

        self.mass = mass
        assert mass > 1e-5
        self.feas_tol = 1e-6
        self.pH_min = pH_min
        self.pH_max = pH_max
        self.vH_min = vH_min
        self.vH_max = vH_max
        self.uH_min = uH_min
        self.uH_max = uH_max
        self.w_ref = w_ref
        self.w_move_to_pR = w_move_to_pR
        self.w_avoid_pR = w_avoid_pR
        self.w_u = w_u
        # self.dt = dt of MPC != dt_ref = dt of ref traj.
        # We will use piecewise cubics to interlate the ref traj.
        self.dt = dt

        self.n_pH = self.pH_min.shape[0]
        self.n_vH = self.vH_min.shape[0]
        self.n_uH = self.uH_min.shape[0]
        # We want the nx0 array, so that tolist works well
        assert self.n_pH == self.n_vH
        assert self.pH_min.shape == (self.n_pH,)
        assert self.pH_max.shape == (self.n_pH,)
        assert self.vH_min.shape == (self.n_vH,)
        assert self.vH_max.shape == (self.n_vH,)
        assert self.uH_min.shape == (self.n_uH,)
        assert self.uH_max.shape == (self.n_uH,)
        assert self.w_ref.shape == (self.n_pH, self.n_pH)
        assert self.w_move_to_pR.shape == (self.n_pH, self.n_pH)
        assert self.w_avoid_pR.shape == (self.n_pH, self.n_pH)
        assert self.w_u.shape == (self.n_uH, self.n_uH)

    def check_shapes(self, pHs_1_T=None, vHs_1_T=None, uHs_1_T=None,
                     pH_0=None, vH_0=None, horizon=None):
        """Ensure all shapes are correct."""
        if pHs_1_T is not None:
            assert pHs_1_T.shape == (horizon, self.n_pH)
        if vHs_1_T is not None:
            assert vHs_1_T.shape == (horizon, self.n_vH)
        if uHs_1_T is not None:
            assert uHs_1_T.shape == (horizon, self.n_uH)
        if pH_0 is not None:
            assert pH_0.shape == (self.n_pH, 1)
        if vH_0 is not None:
            assert vH_0.shape == (self.n_vH, 1)

    def solve_mpc(self, pH_mode, pHs_0_T_ref, dt_ref, vH_0,
                  pR_0=None, plot=False):
        """
        Solve MPC to find a trajectory for the human.

        Parameters
        ----------
        pH_mode: str
                 pH_indep_pR: H-Indep-R condition in the paper.
                 pH_avoid_pR: H-Away-R condition in the paper.
                 pH_move_to_pR: H-To-R condition in the paper.
        pHs_0_T_ref: n_time_stepsxn_pH np array, the reference trajectory.
                     In our work, this comes from a human MDP policy rollout.
        dt_ref: the dt of pHs_0_T_ref.
        vH_0: n_vHx1 np array = initial human velocity.
        pR_0: n_pRx1 np array = initial robot position.
        plot: bool, whether to plot or not (for debugging).

        Returns
        ----------
        pHs_1_T_opt: horizon x n_pH, human positions computed by the MPC.
        vHs_1_T_opt: horizon x n_vH, human velocities computed by the MPC.
        uHs_1_T_opt: horizon x n_uH, controls computed by the MPC.
        """

        assert pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]
        if pH_mode in ["pH_avoid_pR", "pH_move_to_pR"]:
            assert pR_0 is not None
            # pR, pH have to be in the same space
            pH_0 = pHs_0_T_ref[0, :]
            pR_0_sqz = pR_0.squeeze()
            assert pR_0_sqz.shape == pH_0.shape

        pw_spliness, dts_pw = waypts_2_pwsplines(
                wp_traj=pHs_0_T_ref, dt=dt_ref,
                degree=3, plot=False)
        pH_0 = np.reshape(pHs_0_T_ref[0, :], (self.n_pH, 1))
        assert vH_0.shape == (self.n_vH, 1)
        for i in range(self.n_pH):
            tmp = interpolate.splev(x=[0], tck=pw_spliness[i], ext=2)
            assert abs(tmp[0] - pH_0[i, 0]) < 1e-5

        end_time_pwc = dts_pw[-1]
        # Note: horizon + 1 = the length of traj including x0.
        horizon = int(np.ceil(end_time_pwc / self.dt))
        assert horizon * self.dt >= end_time_pwc

        # Decision variables
        pHs_1_T = MX.sym("pHs_1_T", (horizon, self.n_pH))
        vHs_1_T = MX.sym("vHs_1_T", (horizon, self.n_vH))
        uHs_1_T = MX.sym("uHs_1_T", (horizon, self.n_uH))

        # Constraints
        g = []
        lbg = []
        ubg = []
        g_name = []

        # We make terminal goal as in the objective, rather than a constraint.
        terminal_goal_cst = False
        g_bd, lbg_bd, ubg_bd, g_names_bd = self.generate_boundary_csts_bounds(
                pHs_1_T, vHs_1_T, uHs_1_T, horizon, pw_spliness,
                terminal_goal_cst=terminal_goal_cst)
        g = vertcat(g, g_bd)
        lbg += lbg_bd
        ubg += ubg_bd
        g_name += g_names_bd

        # XXX: Collision avoidance with obstacles in the env
        # is handled by the MDP policy.
        g_dyn, lbg_dyn, ubg_dyn, g_names_dyn = self.generate_dyn_csts(
                pHs_1_T, vHs_1_T, uHs_1_T, pH_0, vH_0, horizon)
        g = vertcat(g, g_dyn)
        lbg += lbg_dyn
        ubg += ubg_dyn
        g_name += g_names_dyn

        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)

        track_traj = True
        pTarget = None
        pAvoid = None
        if pH_mode == "pH_indep_pR":
            track_traj = True
            pTarget = None
            pAvoid = None
        elif pH_mode == "pH_move_to_pR":
            track_traj = True
            pTarget = pR_0
            pAvoid = None
        elif pH_mode == "pH_avoid_pR":
            track_traj = True
            pTarget = None
            pAvoid = pR_0
        else:
            raise ValueError()
        cost = self.generate_cost_function(
                pHs_1_T=pHs_1_T, uHs_1_T=uHs_1_T, horizon=horizon,
                pw_spliness=pw_spliness,
                track_traj=track_traj, pTarget=pTarget, pAvoid=pAvoid)

        opt_vars = vertcat(
                pHs_1_T.reshape((-1, 1)),
                vHs_1_T.reshape((-1, 1)),
                uHs_1_T.reshape((-1, 1)))

        prob = {'f': cost, 'x': opt_vars, 'g': g}

        if True:
            opt = {'error_on_fail': False,
                   'ipopt': {
                       'print_level': 0,
                       # 'hessian_approximation': 'limited-memory',
                       "max_iter": 400,
                       "expect_infeasible_problem": "no",
                       "acceptable_tol": 1e-4,
                       "acceptable_constr_viol_tol": 1e-5,
                       "bound_frac": 0.5,
                       "start_with_resto": "no",
                       "required_infeasibility_reduction": 0.85,
                       "acceptable_iter": 8}}  # ipopt
            solver = cas.nlpsol('solver', 'ipopt', prob, opt)
        else:
            raise ValueError()

        # --------------------
        # Solve
        end_time = dts_pw[-1]
        dts = np.linspace(0, end_time, num=horizon+1, endpoint=True)
        # (horizon + 1) x npH
        pHs_0_T_init = np.zeros((dts.shape[0], self.n_pH))
        for i in range(self.n_pH):
            tmp = interpolate.splev(x=dts, tck=pw_spliness[i], ext=2)
            assert pHs_0_T_init[:, i].shape == tmp.shape
            pHs_0_T_init[:, i] = tmp
        pHs_1_T_init = pHs_0_T_init[1:, :]
        assert pHs_1_T_init.shape == pHs_1_T.shape

        pHs_0_T_minus_1_init = pHs_0_T_init[:-1, :]
        uHs_1_T_init = (pHs_1_T_init - pHs_0_T_minus_1_init) / self.dt
        assert uHs_1_T_init.shape == uHs_1_T.shape

        vHs_1_T_init = np.zeros((horizon, self.n_vH))
        assert vHs_1_T_init.shape == vHs_1_T.shape

        opt_vars_init = vertcat(
                pHs_1_T_init.reshape((-1, 1)),
                vHs_1_T_init.reshape((-1, 1)),
                uHs_1_T_init.reshape((-1, 1)))
        assert opt_vars.shape == opt_vars_init.shape

        sol = solver(x0=opt_vars_init, lbg=lbg, ubg=ubg)
        x_opt = sol['x']
        f_opt = sol['f']
        print('f_opt = ', f_opt)

        g_res = np.array(sol["g"]).squeeze()
        feasible = True

        # This is not sufficient to determine feasibility,
        # since casadi gives out wrong feasibility values.
        if np.any(np.array(lbg) - self.feas_tol > g_res) or np.any(
                np.array(ubg) + self.feas_tol < g_res):
            feasible = False
        if not feasible:
            print_FAIL("Hmpc is not feasible")

        # Manually checking feasibility:
        # Get indices of the respective variables:
        c = 0
        n_pHs_1_T = horizon * self.n_pH
        idx_pHs_1_T = np.arange(n_pHs_1_T)
        c += n_pHs_1_T
        n_vHs_1_T = horizon * self.n_vH
        idx_vHs_1_T = np.arange(c, c + n_vHs_1_T)
        c += n_vHs_1_T
        nH_us_1_T = horizon * self.n_uH
        idx_us_1_T = np.arange(c, c + nH_us_1_T)
        c += nH_us_1_T
        assert c == x_opt.shape[0]

        pHs_1_T_opt = np.array(cas_reshape(x_opt[idx_pHs_1_T],
                               (horizon, self.n_pH)))
        assert pHs_1_T.shape == pHs_1_T_opt.shape
        vHs_1_T_opt = np.array(cas_reshape(x_opt[idx_vHs_1_T],
                               (horizon, self.n_vH)))
        assert vHs_1_T.shape == vHs_1_T_opt.shape
        uHs_1_T_opt = np.array(cas_reshape(x_opt[idx_us_1_T],
                               (horizon, self.n_uH)))
        assert uHs_1_T.shape == uHs_1_T_opt.shape

        # ui bound
        if use_human_ui_bound:
            for i in range(horizon - 1):
                ui = uHs_1_T_opt[i, :].T
                if (ui > self.uH_max).any() or (ui < self.uH_min).any():
                    print_FAIL("At i={}, ui={} is out of bound"
                               .format(i, ui))
            uT = uHs_1_T_opt[horizon-1, :].T
            # if (np.abs(uT) > 1e-5).any():
            if (uT > self.uH_max).any() or (uT < self.uH_min).any():
                print_FAIL("At i={}, ui={} is out of bound"
                           .format(horizon-1, uT))
        # pH bound
        for i in range(horizon):
            pH = pHs_1_T_opt[i, :].T
            if (pH > self.pH_max).any() or (pH < self.pH_min).any():
                print_FAIL("At i={}, pH={} is out of bound"
                           .format(i, pH))
        # vH bound
        for i in range(horizon - 1):
            vHi = vHs_1_T_opt[i, :].T
            if (vHi > self.vH_max).any() or (vHi < self.vH_min).any():
                print_FAIL("At i={}, vHi={} is out of bound"
                           .format(i, vHi))
        vHT = vHs_1_T_opt[horizon-1, :].T
        if (np.abs(vHT) > 1e-5).any():
            print_FAIL("At i={}, vHi={} is out of bound"
                       .format(horizon-1, vHT))
        # pH ~ vH
        cur_pH = pH_0
        next_pH = np.reshape(pHs_1_T_opt[0, :].T, (self.n_pH, 1))
        next_vH = np.reshape(vHs_1_T_opt[0, :].T, (self.n_vH, 1))
        assert next_pH.shape == cur_pH.shape == next_vH.shape
        tmp = next_pH - cur_pH - self.dt * next_vH
        if ((np.abs(tmp)) > 1e-5).any():
            print_FAIL("At i={}, cur_pH={}, next_pH={}, next_vH={} out of bound"
                       .format(0, cur_pH, next_pH, next_vH))
        for i in range(horizon - 1):
            cur_pH = pHs_1_T_opt[i, :].T
            next_pH = pHs_1_T_opt[i+1, :].T
            next_vH = vHs_1_T_opt[i+1, :].T
            assert next_pH.shape == cur_pH.shape == next_vH.shape
            tmp = next_pH - cur_pH - self.dt * next_vH
            assert ((np.abs(tmp)) < 1e-5).all()

        # vH ~ uH
        cur_vH = vH_0
        next_vH = np.reshape(vHs_1_T_opt[0, :].T, (self.n_vH, 1))
        next_uH = np.reshape(uHs_1_T_opt[0, :].T, (self.n_uH, 1))
        assert next_vH.shape == cur_vH.shape == next_uH.shape
        tmp = next_vH - cur_vH - self.dt * next_uH / self.mass
        assert ((np.abs(tmp)) < 1e-5).all()
        for i in range(horizon - 1):
            cur_vH = vHs_1_T_opt[i, :].T
            next_vH = vHs_1_T_opt[i+1, :].T
            next_uH = uHs_1_T_opt[i+1, :].T
            assert next_vH.shape == cur_vH.shape == next_uH.shape
            tmp = next_vH - cur_vH - self.dt * next_uH / self.mass
            assert ((np.abs(tmp)) < 1e-5).all()

        # --------------------
        if plot:
            # (horizon + 1) x npH
            pHs_0_T_int = np.zeros((dts.shape[0], self.n_pH))
            for i in range(self.n_pH):
                tmp = interpolate.splev(x=dts, tck=pw_spliness[i], ext=2)
                assert pHs_0_T_int[:, i].shape == tmp.shape
                pHs_0_T_int[:, i] = tmp
            ref = pHs_0_T_ref
            intp = pHs_0_T_int
            opt = np.vstack((pH_0.T, pHs_1_T_opt))

            assert self.n_pH == 2
            fig, ax = plt.subplots()
            ax.plot(opt[:, 0].squeeze(), opt[:, 1].squeeze(), '.', label='opt')
            ax.plot(ref[:, 0].squeeze(), ref[:, 1].squeeze(), 'o', label='ref')
            ax.plot(pH_0.squeeze()[0], pH_0.squeeze()[1], 'x', label='pH_0')
            if pH_mode in ["pH_avoid_pR", "pH_move_to_pR"]:
                ax.plot(pR_0.squeeze()[0], pR_0.squeeze()[1], 'x', label='pR_0')
            ax.legend(loc='upper right', ncol=2)
            plt.show()
            plt.cla()
            plt.clf()
            plt.close()

            fig, ax = plt.subplots()
            ax.plot(dts, opt, 'o', label='opt')
            ax.plot(dts, intp, label="intp_ref")
            ax.legend(loc='upper right', ncol=2)
            plt.show()
            plt.cla()
            plt.clf()
            plt.close()

            for i in range(self.n_pH):
                pHs_0_T_opt = np.vstack(
                        (np.reshape(pH_0, (self.n_pH)), pHs_1_T_opt))
                vHs_0_T_opt = np.vstack(
                        (np.reshape(vH_0, (self.n_vH)), vHs_1_T_opt))
                uHs_0_T_opt = np.vstack((np.zeros(self.n_vH), uHs_1_T_opt))
                # plt.figure()
                fig, axs = plt.subplots(3)
                fig.tight_layout()
                axs[0].plot(dts, pHs_0_T_opt[:, i], 'ro')
                axs[0].plot(dts, pHs_0_T_opt[:, i], 'g')
                axs[0].set_title('pHs_0_T_opt')
                axs[1].plot(dts, vHs_0_T_opt[:, i], 'ro')
                axs[1].plot(dts, vHs_0_T_opt[:, i], 'g')
                axs[1].set_title('vHs_0_T_opt')
                axs[2].plot(dts, uHs_0_T_opt[:, i], 'ro')
                axs[2].plot(dts, uHs_0_T_opt[:, i], 'g')
                axs[2].set_title('uHs_0_T_opt')
                plt.show()
                plt.cla()
                plt.clf()
                plt.close()

        return pHs_1_T_opt, vHs_1_T_opt, uHs_1_T_opt

    def generate_boundary_csts_bounds(self, pHs_1_T, vHs_1_T,
                                      uHs_1_T, horizon,
                                      pw_spliness, terminal_goal_cst):
        """
        Formulate symbolic constraint for the boundary conditions.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH, symbolic human positions.
        vHs_1_T: horizon x n_vH, symbolic human velocities.
        uHs_1_T: horizon x n_uH, symbolic controls.
        horizon: int, horizon of planning.
        pw_spliness: the reference trajectory, used to extract the goal.
        terminal_goal_cst: whether to set goal as a terminal constraint.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pHs_1_T=pHs_1_T, vHs_1_T=vHs_1_T,
                          uHs_1_T=uHs_1_T, horizon=horizon)

        g = []
        lbg = []
        ubg = []
        g_name = []

        # ui bound
        if use_human_ui_bound:
            uH_min = (self.uH_min + self.feas_tol).tolist()
            uH_max = (self.uH_max - self.feas_tol).tolist()
            for i in range(horizon - 1):
                # nuH x 1
                ui = uHs_1_T[i, :].T
                g = vertcat(g, ui)
                lbg += uH_min
                ubg += uH_max
                for j in range(self.n_uH):
                    g_name += ["u_bd_i={}_dim={}".format(i, j)]
                # print("cur={}".format(i))

            # uH terminal = 0
            u_T = uHs_1_T[horizon-1, :].T
            g = vertcat(g, u_T)
            # We need vR terminal = 0, which will be constrained later.
            # So we don't need u terminal to 0.
            lbg += uH_min
            ubg += uH_max
            # lbg += [-1e-9] * self.n_uH
            # ubg += [1e-9] * self.n_uH
            for j in range(self.n_uH):
                g_name += ["u_bd_i={}_dim={}".format(horizon-1, j)]
            # print("cur={}".format(horizon-1))

        # pH bound
        pH_min = (self.pH_min + self.feas_tol).tolist()
        pH_max = (self.pH_max - self.feas_tol).tolist()
        for i in range(horizon - 1):
            # n_pH x 1
            pH = pHs_1_T[i, :].T
            g = vertcat(g, pH)
            lbg += pH_min
            ubg += pH_max
            for j in range(len(pH_min)):
                g_name += ["pH_bd_i={}_dim={}".format(i, j)]
            # print("cur={}".format(i))

        # pH terminal = goal
        pH_T = pHs_1_T[horizon - 1, :].T
        if terminal_goal_cst:
            goal = np.zeros((self.n_pH, 1))
            for i in range(self.n_pH):
                tmp = interpolate.splev(x=[horizon * self.dt],
                                        tck=pw_spliness[i], ext=2)
                assert tmp.shape == goal[i, :].shape
                goal[i, :] = tmp
            g = vertcat(g, goal - pH_T)
            lbg += [-1e-9] * self.n_pH
            ubg += [1e-9] * self.n_pH
            for j in range(self.n_pH):
                g_name += ["pH_bd_goal_i={}_dim={}".format(horizon, j)]
        else:
            g = vertcat(g, pH_T)
            lbg += pH_min
            ubg += pH_max
            for j in range(len(pH_min)):
                g_name += ["pH_bd_i={}_dim={}".format(i, j)]
        # print("cur={}".format(horizon - 1))

        # vH bound
        vH_min = (self.vH_min + self.feas_tol).tolist()
        vH_max = (self.vH_max - self.feas_tol).tolist()
        for i in range(horizon - 1):
            vH = vHs_1_T[i, :].T
            g = vertcat(g, vH)
            lbg += vH_min
            ubg += vH_max
            for j in range(len(vH_min)):
                g_name += ["vH_bd_i={}_dim={}".format(i, j)]
            # print("cur={}".format(i))

        # vH terminal = 0
        vH_T = vHs_1_T[horizon - 1, :].T
        g = vertcat(g, vH_T)
        lbg += [-1e-9] * self.n_vH
        ubg += [1e-9] * self.n_vH
        for j in range(self.n_vH):
            g_name += ["vH_bd_i={}_dim={}".format(horizon-1, j)]
        # print("cur={}".format(self.h_safe - 1))

        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_dyn_csts(self, pHs_1_T, vHs_1_T, uHs_1_T,
                          pH_0, vH_0, horizon):
        """
        Formulate symbolic constraint for the dynamics.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH, symbolic human positions.
        vHs_1_T: horizon x n_vH, symbolic human velocities.
        uHs_1_T: horizon x n_uH, symbolic controls.
        pH_0: n_pHx1, the initial human position.
        vH_0: n_pHx1, the initial human velocity.
        horizon: int, horizon of planning.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pHs_1_T=pHs_1_T, vHs_1_T=vHs_1_T, uHs_1_T=uHs_1_T,
                          pH_0=pH_0, vH_0=vH_0, horizon=horizon)
        g = []
        lbg = []
        ubg = []
        g_name = []

        # pH ~ vH
        cur_pH = pH_0
        next_pH = pHs_1_T[0, :].T
        next_vH = vHs_1_T[0, :].T
        tmp = next_pH - cur_pH - self.dt * next_vH
        assert next_pH.shape == cur_pH.shape == next_vH.shape\
            == tmp.shape == vH_0.shape
        g = vertcat(g, tmp)
        lbg += [-1e-9] * self.n_pH
        ubg += [1e-9] * self.n_pH
        for j in range(self.n_pH):
            g_name += ["pH_dyn_t-1_vs_t0_dim={}".format(j)]
        # print("cur={}, next={}".format(-1, 0))

        for i in range(horizon - 1):
            cur_pH = pHs_1_T[i, :].T
            next_pH = pHs_1_T[i+1, :].T
            next_vH = vHs_1_T[i+1, :].T
            tmp = next_pH - cur_pH - self.dt * next_vH
            assert next_pH.shape == cur_pH.shape == next_vH.shape == tmp.shape
            g = vertcat(g, tmp)
            lbg += [-1e-9] * self.n_pH
            ubg += [1e-9] * self.n_pH
            for j in range(self.n_pH):
                g_name += ["pH_dyn_t{}_vs_t{}_dim={}".format(i, i+1, j)]
            # print("cur={}, next={}".format(i, i+1))

        # vH ~ uH
        cur_vH = vH_0
        next_vH = vHs_1_T[0, :].T
        next_uH = uHs_1_T[0, :].T
        tmp = next_vH - cur_vH - self.dt * next_uH / self.mass
        assert next_uH.shape == next_vH.shape == cur_vH.shape == tmp.shape
        g = vertcat(g, tmp)
        lbg += [-1e-9] * self.n_vH
        ubg += [1e-9] * self.n_vH
        for j in range(self.n_vH):
            g_name += ["vH_dyn_t-1_vs_t0_dim={}".format(j)]
        # print("cur={}, next={}".format(-1, 0))

        for i in range(horizon - 1):
            cur_vH = vHs_1_T[i, :].T
            next_vH = vHs_1_T[i+1, :].T
            next_uH = uHs_1_T[i+1, :].T
            tmp = next_vH - cur_vH - self.dt * next_uH / self.mass
            assert next_uH.shape == next_vH.shape == cur_vH.shape == tmp.shape
            g = vertcat(g, tmp)
            lbg += [-1e-9] * self.n_vH
            ubg += [1e-9] * self.n_vH
            for j in range(self.n_vH):
                g_name += ["vH_dyn_t{}_vs_t{}_dim={}".format(i, i+1, j)]
            # print("cur={}, next={}".format(i, i+1))

        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_coll_avoid_cur_pR_csts(self, pHs_1_T, cur_pR, horizon):
        """
        Formulate symbolic constraint for the collision avoidance constraints.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH, symbolic human positions.
        cur_pR: n_pHx1, current robot position,
                used to produce "human avoiding robot" behavior.
        horizon: int, horizon of planning.

        Returns
        ----------
        g: constraint variables.
        lbg: constraint lower bounds.
        ubg: constraint upper bounds.
        g_name: constraint names.
        """

        self.check_shapes(pHs_1_T=pHs_1_T, pH_0=cur_pR, horizon=horizon)

        g = []
        lbg = []
        ubg = []
        g_name = []

        for t in range(horizon):
            # 2x1
            pH = pHs_1_T[t, :].T
            assert pH.shape == cur_pR.shape
            assert pH.shape[1] == 1
            tmp = pH - cur_pR
            d_sqr = mtimes(tmp.T, tmp)
            g = vertcat(g, d_sqr)
            lbg += [(self.pH_view_pH_pR_min_sep_dist)**2+self.feas_tol]
            ubg += [cas.inf]
            g_name += ["pH_pR_d_i={}".format(t)]
        assert g.shape[0] == len(lbg) == len(ubg) == len(g_name)
        return g, lbg, ubg, g_name

    def generate_cost_function(self, pHs_1_T, uHs_1_T, horizon, pw_spliness,
                               track_traj=True, pTarget=None, pAvoid=None):
        """
        Formulate symbolic cost function.
        1. Stay close to the reference trajectory = pw_spliness.
        2. Control cost.
        3. Stay close to pTarget.
        4. Stay far from pAvoid.

        Parameters
        ----------
        pHs_1_T: horizon x n_pH, symbolic human positions.
        uHs_1_T: horizon x n_uH, symbolic controls.
        horizon: int, horizon of planning.
        pw_spliness: the reference trajectory.
        track_traj: bool, whether to track the reference trajectory or not.
        pTarget: n_pHx1, stay close to pTarget across the entire horizon
        pAvoid: n_pHx1, stay far from pAvoid across the entire horizon

        Returns
        ----------
        cost: symbolic cost variable.
        """

        self.check_shapes(pHs_1_T=pHs_1_T, uHs_1_T=uHs_1_T, horizon=horizon)
        cost = 0

        domain_max_t = pw_spliness[0][0][-1]
        assert domain_max_t == pw_spliness[1][0][-1]

        # Tracking cost
        if track_traj:
            ts = []
            # Note: horizon + 1 = the length of traj including x0.
            for i in range(horizon):
                # Need i+1 here, since t=0 is for pH_0, vH_0 which are fixed.
                t = 0. + self.dt * (i + 1)
                # In case t is beyond the domain of pw_spliness
                if t >= domain_max_t - 1e-9:
                    ts.append(domain_max_t)
                else:
                    ts.append(t)
            # horizon x npH
            pH_refs = np.zeros((horizon, self.n_pH))
            for i in range(self.n_pH):
                tmp = interpolate.splev(x=ts, tck=pw_spliness[i], ext=2)
                assert tmp.shape == pH_refs[:, i].shape
                pH_refs[:, i] = tmp

            # Note: horizon + 1 = the length of traj including x0.
            for i in range(horizon):
                pH_ref = pH_refs[i, :].reshape(self.n_pH, 1)
                pH_t = pHs_1_T[i, :].T
                assert pH_ref.shape == pH_t.shape
                cost += mtimes((pH_ref - pH_t).T,
                               mtimes(self.w_ref, (pH_ref - pH_t)))
                # print("cur={}".format(i))
                assert cost.shape == (1, 1)
            assert t == horizon * self.dt

        # Control cost
        for i in range(horizon):
            uHi = uHs_1_T[i, :].T
            cost += mtimes(uHi.T, mtimes(self.w_u, uHi))
            # print("cur={}".format(i))
            assert cost.shape == (1, 1)

        # Stay close to pTarget
        if pTarget is not None:
            for i in range(horizon):
                pH_t = pHs_1_T[i, :].T
                assert pTarget.shape == pH_t.shape
                cost += mtimes((pTarget - pH_t).T,
                               mtimes(self.w_move_to_pR, (pTarget - pH_t)))
                # print("cur={}".format(i))
                assert cost.shape == (1, 1)

        # Stay far from pAvoid
        if pAvoid is not None:
            for i in range(horizon):
                pH_t = pHs_1_T[i, :].T
                assert pAvoid.shape == pH_t.shape
                cost += - mtimes((pAvoid - pH_t).T,
                                 mtimes(self.w_avoid_pR, (pAvoid - pH_t)))
                # print("cur={}".format(i))
                assert cost.shape == (1, 1)
        return cost


if __name__ == "__main__":
    # Test
    np.random.seed(0)
    # https://stackoverflow.com/a/2891805
    np.set_printoptions(precision=3, suppress=True)

    mpc = HumanRefTracker(
            pH_view_pH_pR_min_sep_dist=0.5,
            pH_min=np.array([-1., -1.], dtype=np.float32),
            pH_max=np.array([4., 4.], dtype=np.float32),
            vH_min=np.array([-100., -100.], dtype=np.float32),
            vH_max=np.array([100., 100.], dtype=np.float32),
            uH_min=np.array([-1000., -1000.], dtype=np.float32),
            uH_max=np.array([1000., 1000.], dtype=np.float32),
            w_ref=np.eye(2)*1., w_u=np.eye(2)*0.0001,
            w_move_to_pR=np.eye(2)*1., dt=0.05, mass=2.,
            w_avoid_pR=np.eye(2)*1.)

    # This is different from the dt in mpc.
    dt_ref = 0.5

    pHs_0_T_ref = np.array([[0, 1], [1, 2], [0, 0], [1, 1], [2, 3]])
    vH_0 = np.array([0, 0], dtype=np.float32)
    vH_0 = vH_0.reshape((2, 1))

    pH_mode = "pH_indep_pR"
    pHs_1_T_opt, vHs_1_T_opt, uHs_1_T_opt = mpc.solve_mpc(
            pH_mode=pH_mode, pHs_0_T_ref=pHs_0_T_ref,
            dt_ref=dt_ref, vH_0=vH_0, pR_0=None, plot=True)

    pH_mode = "pH_avoid_pR"
    pR_0 = np.array([[1, 2]]).T
    pHs_1_T_opt, vHs_1_T_opt, uHs_1_T_opt = mpc.solve_mpc(
            pH_mode=pH_mode, pHs_0_T_ref=pHs_0_T_ref,
            dt_ref=dt_ref, vH_0=vH_0, pR_0=pR_0, plot=True)

    pH_mode = "pH_move_to_pR"
    pR_0 = np.array([[1, 2]]).T
    pHs_1_T_opt, vHs_1_T_opt, uHs_1_T_opt = mpc.solve_mpc(
            pH_mode=pH_mode, pHs_0_T_ref=pHs_0_T_ref,
            dt_ref=dt_ref, vH_0=vH_0, pR_0=pR_0, plot=True)
