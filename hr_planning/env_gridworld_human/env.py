# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''
# Adapted from https://github.com/befelix/safe-exploration/blob/master/safe_exploration/environments.py

import numpy as np
from numpy.matlib import repmat
from scipy.integrate import ode, odeint
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from casadi import reshape as cas_reshape
from matplotlib import cm
import matplotlib.patches as mpatches
import os
import yaml
from scipy import interpolate
import sdeint

import hr_planning
from hr_planning.visualization.utils_visualization import plot_ellipsoid_2D
from hr_planning.env_gridworld_human.hmdp import HumanMdp
from hr_planning.env_gridworld_human.hmpc import HumanRefTracker
from hr_planning.visualization.utils_visualization import print_FAIL, print_OK
from hr_planning.utils_interp import waypts_2_zeroOrderHold


X_LIM_RATIO_VIS = 0.1
Y_LIM_RATIO_VIS = 0.1


class HREnv(object):
    def __init__(self, config_path_hr_env, config_path_hmdp,
                 cache_dir, pH_mode, value_iteration=True, pR_0_arg=None):
        """
        Class for the human robot environment,
        including the human and robot simulators.
        Human dynamics = simulated via sde or ode (depending on self.use_sde).
        Human controls <= human MPC (Hmpc) <= human MDP (Hmdp).
        Robot dynamics = simulated via ode.
        Robot controls <= robot MCP (Rmpc).

        # E.g., in a 2D grid world:
        pH = human position.
        vH = human velocity.
        pR = robot position.
        vR = robot velocity.
        u = aR = robot control.

        State s = pHx, pHy, vHx, vHy, pRx, pRy, vRx, vRy
        sH = pHx, pHy, vHx, vHy
        sR = pRx, pRy, vRx, vRy
        n_s = 8
        n_pR = 2
        n_vR = 2
        n_sR = 4
        n_pH = 2
        n_vH = 2
        n_sH = 4
        u = aRx, aRy
        n_u = 2

        Parameters
        ----------
        config_path_hr_env: str, path for a config file for this class - HREnv.
        config_path_hmdp: str, path for a config file for an Hmdp.
        cache_dir: directory to save Hmdp's computed transition, reward, and policy.
        pH_mode: str
                 pH_indep_pR: H-Indep-R condition in the paper.
                 pH_avoid_pR: H-Away-R condition in the paper.
                 pH_move_to_pR: H-To-R condition in the paper.
        value_iteration: bool, whether to run value iteration
                         or load the policy from files.
        pR_0_arg: (n_pR,) numpy vector, initial robot position
                  If None, then will use the pR_0 from config_path_hr_env.
                  If not None, then will use pR_0_arg.
        """

        assert pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]

        self.pH_mode = pH_mode
        self.iteration = 0
        self.cur_pR = None
        self.cur_vR = None
        self.cur_pH = None
        self.cur_vH = None

        # Used for sdeint
        self.cur_uHs_interp = None
        self.end_time_us_mpc = None

        with open(config_path_hr_env) as f:
            self.config_hr_env = yaml.load(f, Loader=yaml.FullLoader)

        self.mR = self.config_hr_env["mR"]
        self.mH = self.config_hr_env["mH"]

        # dt used for env ode sim
        self.dt_env = self.config_hr_env["dt_env"]
        # dt used for rollout human mdp policy as a ref traj for huma mpc
        self.dt_Hmdp = self.config_hr_env["dt_Hmdp"]
        # dt used for robot mpc
        self.dt_Rmpc = self.config_hr_env["dt_Rmpc"]
        # dt used for higher resolution collision checking
        self.dt_pH_pR_safety_checking\
            = self.config_hr_env["dt_pH_pR_safety_checking"]

        # How often do we update the env, i.e., the amount of time that the
        # function step() will move forward into the future.
        self.step_time = self.config_hr_env["step_time"]

        self.H_sde_noise = self.config_hr_env["H_sde_noise"]
        self.use_sde = self.config_hr_env["use_sde"]

        # Override if needed
        if pR_0_arg is None:
            if "pR_0" in self.config_hr_env:
                self.pR_0 = np.array(self.config_hr_env["pR_0"])
            elif "pR_0_coll_avoid" in self.config_hr_env:
                self.pR_0 = np.array(self.config_hr_env["pR_0_coll_avoid"])
            elif "pR_0_handover" in self.config_hr_env:
                self.pR_0 = np.array(self.config_hr_env["pR_0_handover"])
            else:
                raise RuntimeError()
        else:
            self.pR_0 = np.array(pR_0_arg)
        self.vR_0 = np.array(self.config_hr_env["vR_0"])

        self.n_pR = self.pR_0.shape[0]
        self.n_vR = self.n_pR
        self.n_uR = self.n_pR

        self.pR_0 = self.pR_0.reshape((self.n_pR, 1))
        self.vR_0 = self.vR_0.reshape((self.n_vR, 1))

        self.pR_min = np.array(self.config_hr_env["pR_min"])
        self.pR_min = self.pR_min.reshape((self.n_pR, 1))
        self.pR_max = np.array(self.config_hr_env["pR_max"])
        self.pR_max = self.pR_max.reshape((self.n_pR, 1))

        self.vR_min = np.array(self.config_hr_env["vR_min"])
        self.vR_min = self.vR_min.reshape((self.n_vR, 1))
        self.vR_max = np.array(self.config_hr_env["vR_max"])
        self.vR_max = self.vR_max.reshape((self.n_vR, 1))

        self.uR_min = np.array(self.config_hr_env["uR_min"])
        self.uR_min = self.uR_min.reshape((self.n_uR, 1))
        self.uR_max = np.array(self.config_hr_env["uR_max"])
        self.uR_max = self.uR_max.reshape((self.n_uR, 1))

        self.pR_goal = np.array(self.config_hr_env["pR_goal"])
        self.pR_goal = self.pR_goal.reshape((self.n_pR, 1))
        self.pR_goal_tol = self.config_hr_env["pR_goal_tol"]

        assert (self.pR_0 < 1e-5 + self.pR_max).all()
        assert (self.pR_0 > -1e-5 + self.pR_min).all()
        assert (self.vR_0 < 1e-5 + self.vR_max).all()
        assert (self.vR_0 > -1e-5 + self.vR_min).all()
        assert (self.pR_goal < 1e-5 + self.pR_max).all()
        assert (self.pR_goal > -1e-5 + self.pR_min).all()

        self.Hmdp = HumanMdp(config_path=config_path_hmdp,
                             cache_dir=cache_dir)
        if value_iteration:
            self.Hmdp.computeTransitionAndRewardArrays()
            self.Hmdp.value_iteration(discount=1, epsilon=1e-5, max_iter=10000)
            self.Hmdp.printPolicy()
        self.Hmdp.loadTransitionAndRewardArrays()
        self.Hmdp.loadPolicyVI()

        self.n_pH = self.Hmdp.ss.n_dofs
        self.n_vH = self.n_pH
        self.n_uH = self.n_pH

        self.pH_0 = np.array(self.config_hr_env["pH_0"])
        self.pH_0 = self.pH_0.reshape((self.n_pH, 1))
        self.vH_0 = np.array(self.config_hr_env["vH_0"])
        self.vH_0 = self.vH_0.reshape((self.n_vH, 1))

        # For the dressing task
        if "pH_shoulder" in self.config_hr_env:
            self.pH_shoulder = np.array(self.config_hr_env["pH_shoulder"])
            self.pH_shoulder = self.pH_shoulder.reshape((self.n_pH, 1))
            self.n_interp_pH_pH_shoulder = int(
                    self.config_hr_env["n_interp_pH_pH_shoulder"])
            self.max_dist_bw_pR_arm = self.config_hr_env["max_dist_bw_pR_arm"]

        self.pH_min = np.zeros((self.n_pH, 1))
        self.pH_max = np.zeros((self.n_pH, 1))
        for i in range(self.n_pH):
            cs = self.Hmdp.ss.ind_2_center_by_dof[i]
            r = self.Hmdp.ss.ind_2_radii_by_dof[i]
            assert abs(cs[1] - cs[0] - r * 2) < 1e-5
            self.pH_min[i] = cs[0] - r
            self.pH_max[i] = cs[-1] + r

        self.pH_min = self.pH_min.reshape((self.n_pH, 1))
        self.pH_min = self.pH_min.reshape((self.n_pH, 1))
        self.vH_min = np.array(self.config_hr_env["Hmpc"]["vH_min"])
        self.vH_min = self.vH_min.reshape((self.n_vH, 1))
        self.vH_max = np.array(self.config_hr_env["Hmpc"]["vH_max"])
        self.vH_max = self.vH_max.reshape((self.n_vH, 1))
        self.uH_min = np.array(self.config_hr_env["Hmpc"]["uH_min"])
        self.uH_min = self.uH_min.reshape((self.n_uH, 1))
        self.uH_max = np.array(self.config_hr_env["Hmpc"]["uH_max"])
        self.uH_max = self.uH_max.reshape((self.n_uH, 1))

        assert (self.pH_0 < 1e-5 + np.squeeze(self.pH_max)).all()
        assert (self.pH_0 > -1e-5 + np.squeeze(self.pH_min)).all()
        assert (self.vH_0 < 1e-5 + np.squeeze(self.vH_max)).all()
        assert (self.vH_0 > -1e-5 + np.squeeze(self.vH_min)).all()

        # For workspace constraint
        if "pH_min_workspace" not in self.config_hr_env:
            self.pH_min_workspace = np.copy(self.pH_min)
            self.pH_max_workspace = np.copy(self.pH_max)
        else:
            self.pH_min_workspace = np.array(self.config_hr_env["pH_min_workspace"])
            self.pH_max_workspace = np.array(self.config_hr_env["pH_max_workspace"])
            self.pH_min_workspace = self.pH_min_workspace.reshape((self.n_pH, 1))
            self.pH_max_workspace = self.pH_max_workspace.reshape((self.n_pH, 1))

        self.w_ref = np.eye(self.n_pH)*self.config_hr_env["Hmpc"]["w_ref"]
        self.w_u = np.eye(self.n_uH)*self.config_hr_env["Hmpc"]["w_u"]
        self.w_move_to_pR = np.eye(self.n_pH)*self.config_hr_env["Hmpc"]["w_move_to_pR"]
        self.w_avoid_pR = np.eye(self.n_pH)*self.config_hr_env["Hmpc"]["w_avoid_pR"]
        # dt used for human mpc
        self.dt_Hmpc = self.config_hr_env["Hmpc"]["dt_Hmpc"]

        # This is from robot MPC (Rmpc)'s view, how far to avoid pH
        self.pH_pR_min_sep_dist = float(
                self.config_hr_env["pH_pR_min_sep_dist"])

        # This is from human MPC (Hmpc)'s view, how far to avoid pR_0
        pH_view_pH_pR_min_sep_dist = float(
                self.config_hr_env["Hmpc"]["pH_view_pH_pR_min_sep_dist"])
        self.Hmpc = HumanRefTracker(
                 pH_min=np.squeeze(self.pH_min),
                 pH_max=np.squeeze(self.pH_max),
                 vH_min=np.squeeze(self.vH_min),
                 vH_max=np.squeeze(self.vH_max),
                 uH_min=np.squeeze(self.uH_min),
                 uH_max=np.squeeze(self.uH_max),
                 w_ref=self.w_ref, w_u=self.w_u,
                 w_move_to_pR=self.w_move_to_pR,
                 w_avoid_pR=self.w_avoid_pR,
                 dt=self.dt_Hmpc, mass=self.mH,
                 pH_view_pH_pR_min_sep_dist=pH_view_pH_pR_min_sep_dist)

        self.color_robot = "r"
        self.color_human = "g"
        self.color_hmdp_grid = "k"
        # https://matplotlib.org/3.3.3/api/markers_api.html
        self.marker_hmdp_obstacle = "x"
        self.marker_hmdp_grid = "."
        self.marker_robot_init_state = "X"
        self.marker_human_init_state = "X"
        self.marker_robot_traj = "s"
        self.marker_human_traj = "s"
        self.marker_human_data = "o"
        self.alpha_hmdp_grid = 0.1
        self.alpha_hmdp_obstacle = 0.1
        self.markersize_state = 10
        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        self.ellipsoid_cmap = cm.get_cmap('Paired')

        # ---------------------
        # Collision checking
        # We approximate the robot collision volume as a set of pts
        # to ease the collision checking wrt human as ellipsoids.
        # In MPC, we will add the offsets to pR and check whether
        # that point is inside or not the ellipsoids.
        self.R_col_volume_offsets = np.array([[0, 0]]) # only check collision at pR without any offsets.
        assert self.R_col_volume_offsets.shape[1] == self.n_pR

        # Safe impact potential
        assert self.n_vR == self.n_vH
        # Worst case = 1, based on <Quanti...> paper
        self.coeff_restit = self.config_hr_env["coeff_restit"]
        self.F_HR_max_safe_impact = self.config_hr_env["F_HR_max_safe_impact"]
        # Convert it to inscribed hyperrectangle, along x,y
        # nvRx0
        self.F_HR_max = np.zeros(self.n_vR,)
        self.F_HR_max.fill(self.F_HR_max_safe_impact / np.sqrt(self.n_vR))

        # Eq6 in paper:
        # 1x1
        tmp = abs((self.coeff_restit + 1.) / (1./self.mR + 1./self.mH))
        assert tmp > 1e-5
        # \rho in App.D(A):
        # nvRx0
        self.safe_pot_offset = self.F_HR_max / tmp

        # Eq.11 in paper: Cst: h_mat x vH <= [-vR, vR]^T + h
        # nvRx1
        tmp2 = np.reshape(self.safe_pot_offset, (self.n_vR, 1))
        # (2*nvR)x1
        self.h_safe_imp_pot = np.vstack((tmp2, tmp2))
        # (2*nvR)xnvR
        self.h_mat_safe_imp_pot = np.vstack(
                (-np.eye(self.n_vR), np.eye(self.n_vR)))

    def noise_H_sde(self, x, t):
        """Definition of the noise for human sde"""
        # https://pypi.org/project/sdeint/
        # diagonal, so independent driving Wiener processes
        pH_noise = [self.H_sde_noise] * self.n_pH
        vH_noise = [0.] * self.n_vH
        return np.diag(pH_noise + vH_noise)

    def dynamics_H_sde(self, y, t):
        """Definition of the dynamics for human sde.
        sdeint does not allow passing in args"""
        return self.dynamics(
                y=y, t=t, uts=self.cur_uHs_interp,
                end_time=self.end_time_us_mpc, m=self.mH)

    def dynamics(self, y, t, uts, end_time, m):
        """Definition of the double integrator dynamics
        (used for human ode and robot ode).

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
        Computes the derivative of y at t.

        Parameters
        ----------
        y: (np+nv)x0 = state s = [px, py, vx, vy].
        t: float, the current time.
        uts: scipy.interpolate.interpolate.interp1d = controls across time.
        end_time: the end time of uts.
        m: mass.

        Returns
        -------
        dy: (np+nv)x1.
        """
        ut = np.zeros((uts.y.shape[1],))
        if t <= end_time:
            ut = uts(t)

        n_x = y.shape[0]
        n_p = int(n_x / 2.)
        dy = np.zeros((n_x,))
        assert n_x / 2 == 2
        dy[:n_p] = y[2:]
        dy[n_p:] = ut / m

        return dy

    def sample_init_sR(self, mean, std, n_samples=1):
        """Sample an initial robot state, sR.
        mean: nsx0.
        std: float.
        n_samples: int.
        """
        n_s = mean.shape[0]
        samples = (repmat(std, n_samples, 1)
                   * np.random.randn(n_samples, n_s)
                   + repmat(mean, n_samples, 1))
        return (samples.T.squeeze()).T

    def random_uR(self):
        """Sample a random control, uR."""
        return np.random.rand(self.n_uR)\
            * (self.uR_max - self.uR_min) + self.uR_min

    def reset(self, pR_0=None, vR_0=None, pH_0=None, vH_0=None):
        """
        Reset the env to:
        pR_0: initial robot position.
        vR_0: initial robot velocity.
        pH_0: initial human position.
        vH_0: initial human velocity.
        """
        self.iteration = 0
        if pR_0 is None:
            pR_0 = self.pR_0
        if vR_0 is None:
            vR_0 = self.vR_0
        if pH_0 is None:
            pH_0 = self.pH_0
        if vH_0 is None:
            vH_0 = self.vH_0
        assert (pR_0 < 1e-5 + np.squeeze(self.pR_max)).all()
        assert (pR_0 > -1e-5 + np.squeeze(self.pR_min)).all()
        assert (vR_0 < 1e-5 + np.squeeze(self.vR_max)).all()
        assert (vR_0 > -1e-5 + np.squeeze(self.vR_min)).all()
        assert (pH_0 < 1e-5 + np.squeeze(self.pH_max)).all()
        assert (pH_0 > -1e-5 + np.squeeze(self.pH_min)).all()
        assert (vH_0 < 1e-5 + np.squeeze(self.vH_max)).all()
        assert (vH_0 > -1e-5 + np.squeeze(self.vH_min)).all()

        self.cur_pR = pR_0
        self.cur_vR = vR_0
        self.cur_pH = pH_0
        self.cur_vH = vH_0

        return self.cur_pR, self.cur_vR, self.cur_pH, self.cur_vH

    def sim_ode(self, cur_p, cur_v, us_1_T, dt_us, human_or_robot, plot=False):
        """
        Simulate human/robot ode, and human sde for 1 step.

        Parameters
        ----------
        cur_p: current position.
        cur_v: current velocity.
        us_1_T: controls from time index 1 to T.
        dt_us: the dt associated for us_1_T.
        human_or_robot: str, "R" OR "H".
                        If "H": this is called for simulating the human ode+sde.
                        If "R": this is called for simulating the robot ode.
        plot: bool, plot for debugging.

        Returns
        -------
        next_p: next position from the ode.
        next_v: next velocity from the ode.
        next_p_sde: next velocity from the sde.
        next_v_sde: next velocity from the sde.
        ax: Axes object, containing the plot.
        """

        assert human_or_robot in ["R", "H"]
        cur_p = np.squeeze(cur_p)
        cur_v = np.squeeze(cur_v)
        n_s = cur_p.shape[0]

        # us_1_T = [1:T], so its length = horizon
        horizon_us_mpc = us_1_T.shape[0]
        # dts_us_mpc = [0:T-dt_us]
        dts_us_mpc, step_uHs_mpc = np.linspace(
                start=0.0, stop=horizon_us_mpc*dt_us - dt_us,
                num=horizon_us_mpc, endpoint=True, retstep=True)
        u_interp = waypts_2_zeroOrderHold(wp_traj=us_1_T, dt=dt_us,
                                          axis=0, plot=False)
        end_time_us_mpc = horizon_us_mpc*dt_us - dt_us
        # Integrate 1 step = self.step_time
        # dts_ode = 0:self.step_time
        horizon_ode = int(np.ceil(self.step_time / self.dt_env)) + 1
        dts_ode, step_ode = np.linspace(
                start=0., stop=self.step_time,
                num=horizon_ode, endpoint=True, retstep=True)
        assert step_ode <= self.dt_env

        horizon_ode = int(np.ceil(self.step_time / self.dt_env)) + 1
        dts_ode, step_ode = np.linspace(
                start=0., stop=self.step_time,
                num=horizon_ode, endpoint=True, retstep=True)
        assert step_ode <= self.dt_env
        # Rollout the traj by Hmpc in ode
        mass = None
        if human_or_robot == "H":
            mass = self.mH
        else:
            mass = self.mR
        y0 = np.squeeze(np.hstack((cur_p, cur_v)))
        sol = odeint(func=self.dynamics, y0=y0, t=dts_ode,
                     args=(u_interp, end_time_us_mpc, mass))
        if human_or_robot == "H":
            self.cur_uHs_interp = u_interp
            self.end_time_us_mpc = end_time_us_mpc
            # https://pypi.org/project/sdeint/
            # https://stackoverflow.com/questions/54532246/how-to-implement-a-system-of-stochastic-odes-sdes-in-python
            # https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
            sol_sde = sdeint.itoint(f=self.dynamics_H_sde,
                                    G=self.noise_H_sde,
                                    y0=y0, tspan=dts_ode)
        next_p = sol[-1, :n_s].reshape((n_s, 1))
        next_v = sol[-1, n_s:].reshape((n_s, 1))
        next_p_sde = None
        next_v_sde = None
        if human_or_robot == "H":
            next_p_sde = sol_sde[-1, :n_s].reshape((n_s, 1))
            next_v_sde = sol_sde[-1, n_s:].reshape((n_s, 1))

        ax = None
        if plot:
            # Plot beyond self.step_time
            end_time = dts_us_mpc[-1] + dt_us
            horizon_full_ode = int(np.ceil(end_time / self.dt_env)) + 1
            dts_full_ode, step_full_ode = np.linspace(
                    start=0., stop=end_time, num=horizon_full_ode,
                    endpoint=True, retstep=True)
            assert step_full_ode <= self.dt_env

            full_sol = odeint(func=self.dynamics, y0=y0, t=dts_full_ode,
                              args=(u_interp, end_time_us_mpc, mass))
            if human_or_robot == "H":
                full_sol_sde = sdeint.itoint(f=self.dynamics_H_sde,
                                             G=self.noise_H_sde,
                                             y0=y0, tspan=dts_full_ode)
            fig, ax = plt.subplots()
            ax.plot(dts_full_ode, full_sol[:, :n_s], label='ode')
            ax.plot(dts_ode, sol[:, :n_s], label='partial ode',
                    linewidth=7, alpha=0.4)
            if human_or_robot == "H":
                ax.plot(dts_full_ode, full_sol_sde[:, :n_s], label='sde')
                ax.plot(dts_ode, sol_sde[:, :n_s], label='partial sde',
                        linewidth=7, alpha=0.4)
            # ax.legend(loc='lower right', ncol=2)
            # plt.show()

        return next_p, next_v, next_p_sde, next_v_sde, ax

    def step_human(self, cur_pR, cur_vR, cur_pH, cur_vH, plot=False):
        """
        Simulate the human Hmpc (Hmdp) and ode/sde for 1 step,
        by calling Hmpc (Hmdp) and then sim_ode().

        Parameters
        ----------
        cur_pR: current robot position.
        cur_vR: current robot velocity.
        cur_pH: current human position.
        cur_vH: current human velocity.
        plot: bool, plot for debugging.

        Returns
        -------
        next_pH: next human position from the ode.
        next_vH: next human velocity from the ode.
        next_pH_sde: next human velocity from the sde.
        next_vH_sde: next human velocity from the sde.
        """

        cur_pH = np.reshape(cur_pH, (self.n_pH,))
        assert (cur_pH < np.squeeze(self.pH_max)).all()
        assert (cur_pH > np.squeeze(self.pH_min)).all()

        # Rollout till human reaches goal
        ind0 = self.Hmdp.ss.positions2Ind(tuple(cur_pH))
        # Ensure that ind_traj is longer than self.step_time.
        # i.e, we want traj with length >= min_horizon+1,
        # which represents the time index = 0:min_horizon
        min_horizon = int(np.ceil(self.step_time / self.dt_Hmdp))
        # If len of ref traj < 4, cubic splines will fail
        min_horizon = max(min_horizon, 3)

        ind_traj = self.Hmdp.rollout_ind_traj(
                ind0=ind0, min_horizon=min_horizon)
        assert (len(ind_traj) >= min_horizon + 1)
        assert ind0 in ind_traj
        pHs_0_T_ref = np.zeros((len(ind_traj), self.n_pH))
        for t, indt in enumerate(ind_traj):
            pHs_0_T_ref[t, :] = self.Hmdp.ss.ind2Positions(indt)
        # pHs_0_T_ref[0] is the center of the grid at t=0.
        # However, at t=0, human is at cur_pH.
        # So need to adjust that:
        pHs_0_T_ref[0, :] = cur_pH
        assert (pHs_0_T_ref[0] == cur_pH).all()
        for t in range(pHs_0_T_ref.shape[0]):
            assert (pHs_0_T_ref[t, :] < np.squeeze(self.pH_max)).all()
            assert (pHs_0_T_ref[t, :] > np.squeeze(self.pH_min)).all()
        # pHs_0_T_ref = 0:T, so its length = horizon + 1
        horizon_ref_hmdp = pHs_0_T_ref.shape[0] - 1
        dt_ref_hmdp = self.dt_Hmdp
        dts_ref_hmdp, step_ref_hmdp = np.linspace(
                start=0.0, stop=horizon_ref_hmdp*dt_ref_hmdp,
                num=horizon_ref_hmdp + 1, endpoint=True, retstep=True)
        assert dts_ref_hmdp[0] == 0.
        assert dts_ref_hmdp[-1] >= self.step_time

        # MPC
        # If uHs_1_T_opt.shape[0] <= 1, cubic splines will fail
        assert pHs_0_T_ref.shape[0] >= 4
        pHs_1_T_opt, vHs_1_T_opt, uHs_1_T_opt = self.Hmpc.solve_mpc(
                pHs_0_T_ref=pHs_0_T_ref, dt_ref=self.dt_Hmdp,
                vH_0=cur_vH, pR_0=cur_pR,
                pH_mode=self.pH_mode, plot=False)

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pHs_0_T_ref[:, 0], pHs_0_T_ref[:, 1], 'g', label='ref')
        ax.plot(pHs_1_T_opt[:, 0], pHs_1_T_opt[:, 1], 'r', label='opt')
        ax.plot(cur_pR[0, 0], cur_pR[1, 0], 'bx', label='cur_pR')
        ax.plot(cur_pH[0], cur_pH[1], 'rx', label='cur_pH')
        ax.legend(loc='upper right', ncol=2)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        for i in range(pHs_1_T_opt.shape[0]):
            dist = np.linalg.norm(pHs_1_T_opt[i, :] - cur_pH, ord=2)
            print(dist)
        '''

        # If uHs_1_T_opt.shape[0] <= 1, splines will fail
        assert uHs_1_T_opt.shape[0] > 1
        next_pH, next_vH, next_pH_sde, next_vH_sde, ax = self.sim_ode(
                cur_p=cur_pH, cur_v=cur_vH, us_1_T=uHs_1_T_opt,
                dt_us=self.dt_Hmpc, human_or_robot="H", plot=plot)
        if plot:
            ax.plot(dts_ref_hmdp, pHs_0_T_ref, 'o',
                    markersize=10, label='Hmdp')

            horizon_us_mpc = uHs_1_T_opt.shape[0]
            dt_us = self.dt_Hmpc
            dts_us_mpc, step_uHs_mpc = np.linspace(
                    start=0.0, stop=horizon_us_mpc*dt_us - dt_us,
                    num=horizon_us_mpc, endpoint=True, retstep=True)
            dts_mpc = np.hstack((dts_us_mpc, dts_ref_hmdp[-1]))
            pHs_0_T_opt = np.vstack((cur_pH.T, pHs_1_T_opt))
            ax.plot(dts_mpc, pHs_0_T_opt, label='Hmpc')
            ax.legend(loc='lower right', ncol=2)
            plt.show()

        return next_pH, next_vH, next_pH_sde, next_vH_sde

    def step_robot(self, cur_pR, cur_vR, cur_pH, cur_vH, uRs_1_T, plot=False):
        """
        Simulate the robot ode for 1 step, by calling sim_ode().

        Parameters
        ----------
        cur_pR: current robot position.
        cur_vR: current robot velocity.
        cur_pH: current human position.
        cur_vH: current human velocity.
        uRs_1_T: robot controls from time step 1 to T.
        plot: bool, plot for debugging.

        Returns
        -------
        next_pR: next robot position from the ode.
        next_vR: next robot velocity from the ode.
        """

        assert uRs_1_T.shape[1] == self.n_uR

        # Ensure that uRs_1_T is longer than self.step_time.
        # i.e, we want uRs_1_T with length >= min_horizon,
        # which represents the time index = 0+dt_Rmpc : min_horizon
        min_horizon = int(np.ceil(self.step_time / self.dt_Rmpc))
        tmp = np.zeros((min_horizon, self.n_uR), dtype=np.float32)
        for i in range(min(min_horizon, uRs_1_T.shape[0])):
            tmp[i, :] = uRs_1_T[i, :]
        uRs_1_T = tmp
        assert (uRs_1_T.shape[0] == min_horizon)
        # If uRs_1_T.shape[0] <= 1, splines will fail
        if uRs_1_T.shape[0] <= 1:
            uRs_1_T = np.vstack((uRs_1_T, np.zeros(self.n_uR,)))
        assert uRs_1_T.shape[0] > 1

        next_pR, next_vR, _, _, ax = self.sim_ode(
                cur_p=cur_pR, cur_v=cur_vR, us_1_T=uRs_1_T,
                dt_us=self.dt_Rmpc, human_or_robot="R", plot=plot)
        if plot:
            ax.legend(loc='lower right', ncol=2)
            plt.show()

        return next_pR, next_vR

    def step(self, uRs_1_T, cur_pR=None, cur_vR=None, cur_pH=None,
             cur_vH=None, set_cur_state=True):
        """
        1. Simulate the human Hmpc (Hmdp) and ode/sde for 1 step,
            by calling Hmpc (Hmdp) and then sim_ode().
        2. Simulate the robot ode for 1 step, by calling sim_ode().
        3. Check safety metrics between human and robot.

        Parameters
        ----------
        uRs_1_T: robot controls from time step 1 to T.
        cur_pR: current robot position.
        cur_vR: current robot velocity.
        cur_pH: current human position.
        cur_vH: current human velocity.
        set_cur_state: bool, whether to set the class member variables
                       regarding the current state.

        Returns
        -------
        next_pH: next human position from the ode/sde.
        next_vH: next human velocity from the ode/sde.
        next_pR: next robot position from the ode.
        next_vR: next robot velocity from the ode.
        collision: bool, whether human and robot are in collision in this step.
        safe_impact: bool, whether human and robot have safe impact in this step.
        HR_min_dist: float, min separation distance between human and robot.
        HR_max_vel_diff: float, max velocity difference between human and robot.
        """

        if cur_pR is None:
            cur_pR = self.cur_pR
        if cur_vR is None:
            cur_vR = self.cur_vR
        if cur_pH is None:
            cur_pH = self.cur_pH
        if cur_vH is None:
            cur_vH = self.cur_vH
        assert (cur_pR < 1e-5 + np.squeeze(self.pR_max)).all()
        assert (cur_pR > -1e-5 + np.squeeze(self.pR_min)).all()
        assert (cur_vR < 1e-5 + np.squeeze(self.vR_max)).all()
        assert (cur_vR > -1e-5 + np.squeeze(self.vR_min)).all()
        assert (cur_pH < 1e-5 + np.squeeze(self.pH_max)).all()
        assert (cur_pH > -1e-5 + np.squeeze(self.pH_min)).all()
        assert (cur_vH < 1e-5 + np.squeeze(self.vH_max)).all()
        assert (cur_vH > -1e-5 + np.squeeze(self.vH_min)).all()

        # Step human
        next_pH, next_vH, _next_pH_sde, _next_vH_sde = self.step_human(
                cur_pR=cur_pR, cur_vR=cur_vR,
                cur_pH=cur_pH, cur_vH=cur_vH, plot=False)

        # Step robot
        next_pR, next_vR = self.step_robot(
                cur_pR=cur_pR, cur_vR=cur_vR, cur_pH=cur_pH,
                cur_vH=cur_vH, uRs_1_T=uRs_1_T, plot=False)

        if self.use_sde:
            next_pH = _next_pH_sde
            next_vH = _next_vH_sde

        # Need clipping since ode might out of bound.
        tol = 1e-5
        next_pH = np.clip(next_pH, self.pH_min+tol, self.pH_max-tol)
        next_vH = np.clip(next_vH, self.vH_min+tol, self.vH_max-tol)
        next_pR = np.clip(next_pR, self.pR_min+tol, self.pR_max-tol)
        next_vR = np.clip(next_vR, self.vR_min+tol, self.vR_max-tol)

        print("BEFORE: pR={}".format(np.squeeze(cur_pR)))
        print("AFTER: pR={}".format(np.squeeze(next_pR)))
        print("BEFORE: vR={}".format(np.squeeze(cur_vR)))
        print("AFTER: vR={}".format(np.squeeze(next_vR)))
        print("BEFORE: pH={}".format(np.squeeze(cur_pH)))
        print("AFTER: pH={}".format(np.squeeze(next_pH)))
        print("BEFORE: vH={}".format(np.squeeze(cur_vH)))
        print("AFTER: vH={}".format(np.squeeze(next_vH)))

        collision, safe_impact, HR_min_dist, HR_max_vel_diff\
            = self.check_safety_interp(
                    cur_pR=cur_pR, cur_vR=cur_vR,
                    cur_pH=cur_pH, cur_vH=cur_vH,
                    next_pR=next_pR, next_vR=next_vR,
                    next_pH=next_pH, next_vH=next_vH)
        if collision:
            if safe_impact:
                print_FAIL("Collision && safe impact")
                print_OK("HR_min_dist={}".format(HR_min_dist))
                print_OK("HR_max_vel_diff={}".format(HR_max_vel_diff))
            else:
                print_FAIL("Collision && UNsafe impact")
                print_FAIL("HR_min_dist={}".format(HR_min_dist))
                print_FAIL("HR_max_vel_diff={}".format(HR_max_vel_diff))
        else:
            print_FAIL("SAFE")
            print("HR_min_dist={}".format(HR_min_dist))
            print("HR_max_vel_diff={}".format(HR_max_vel_diff))
            print("safe_pot_offset={}".format(
                np.squeeze(self.safe_pot_offset)))

        if set_cur_state:
            self.iteration += 1
            self.cur_pR = next_pR
            self.cur_vR = next_vR
            self.cur_pH = next_pH
            self.cur_vH = next_vH

        return next_pH, next_vH, next_pR, next_vR,\
            collision, safe_impact, HR_min_dist, HR_max_vel_diff

    def plot_safety_bounds(self, ax=None, plot_human_grid=False,
                           plot_world_rectangle=True):
        """
        Given a 2D axes object, plot the safety bounds on it.

        Parameters
        ----------
        ax: Axes object,
            The input axes object to plot on
        Returns
        -------
        ax: Axes object
            The same Axes object as the input ax but now contains the rectangle
        """

        new_ax = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            new_ax = True
        plt.sca(ax)

        # Create a Rectangle patch
        rect_length1 = (self.pR_max - self.pR_min)[0]
        rect_length2 = (self.pR_max - self.pR_min)[1]
        rect = patches.Rectangle(
                (tuple(self.pR_min)), rect_length1, rect_length2,
                linewidth=1, edgecolor=self.color_robot, facecolor='none')
        if plot_world_rectangle:
            ax.add_patch(rect)

        rect_length3 = (self.pH_max - self.pH_min)[0]
        rect_length4 = (self.pH_max - self.pH_min)[1]
        rect2 = patches.Rectangle(
                (tuple(self.pH_min)), rect_length3, rect_length4,
                linewidth=1, edgecolor=self.color_human, facecolor='none')
        if plot_world_rectangle:
            ax.add_patch(rect2)

        if plot_human_grid:
            assert self.n_pH == 2
            xs = []
            ys = []
            xs_obs = []
            ys_obs = []
            for s in range(self.Hmdp.ss.n_states):
                positions = self.Hmdp.ss.ind2Positions(s)
                if s not in self.Hmdp.inds_obstacle:
                    xs.append(positions[0])
                    ys.append(positions[1])
                else:
                    xs_obs.append(positions[0])
                    ys_obs.append(positions[1])
            plt.scatter(xs, ys, alpha=self.alpha_hmdp_grid,
                        c=self.color_hmdp_grid,
                        marker=self.marker_hmdp_grid)
            plt.scatter(xs_obs, ys_obs, alpha=self.alpha_hmdp_obstacle,
                        c=self.color_hmdp_grid,
                        marker=self.marker_hmdp_obstacle)
        if new_ax:
            x_padding = max(rect_length1, rect_length3) * X_LIM_RATIO_VIS
            y_padding = max(rect_length2, rect_length4) * Y_LIM_RATIO_VIS
            min_x_lim = min(self.pH_min[0], self.pR_min[0]) - x_padding
            max_x_lim = max(self.pH_max[0], self.pR_max[0]) + x_padding
            min_y_lim = min(self.pH_min[1], self.pR_min[1]) - y_padding
            max_y_lim = max(self.pH_max[1], self.pR_max[1]) + y_padding
            ax.set_xlim(min_x_lim, max_x_lim)
            ax.set_ylim(min_y_lim, max_y_lim)
            return fig, ax
        return ax

    def plot_state(self, ax, x, color="b", label="", alpha=1.0, annotate=True,
                   marker="o", markersize_state=1.):
        """
        Plot a given state vector

        Parameters:
        -----------
        ax: Axes Object
            The axes to plot the state on
        x: 2x0 array_like[float], optional
            A state vector of the dynamics
        Returns
        -------
        ax: Axes Object
            The axes with the state plotted
        """
        n_s = x.shape[0]
        assert x.shape == (n_s,)
        plt.sca(ax)
        handles = ax.plot(x[0], x[1], color=color, marker=marker,
                          markersize=markersize_state, alpha=alpha)
        if annotate:
            ax.annotate(label, (x[0], x[1]))
        return ax, handles

    def plot_traj(self, traj, human_or_robot, ax=None, annotate=True):
        """
        Plot a trajectory of states, either of human or robot

        Parameters
        ----------
        traj: trajectory of states.
        ax: Axes Object
            The axes to plot the state on
        Returns
        -------
        ax: Axes Object
            The axes with the trajectory plotted
        """

        assert human_or_robot in ["H", "R"]
        if human_or_robot == "H":
            n_s = self.n_pH
            color = self.color_human
            marker_init_state = self.marker_human_init_state
            marker_traj = self.marker_human_traj
            txt = "H"
        else:
            n_s = self.n_pR
            color = self.color_robot
            marker_init_state = self.marker_robot_init_state
            marker_traj = self.marker_robot_traj
            txt = "R"
        assert traj.shape[1] == n_s
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        plt.sca(ax)

        horizon = traj.shape[0]
        handles = [None] * horizon
        for i in range(horizon):
            pt = traj[i, :]
            if i == 0:
                ax, handles[i] = self.plot_state(
                        ax=ax, x=pt, color=color, label=txt+str(i),
                        annotate=annotate,
                        marker=marker_init_state, alpha=1.,
                        markersize_state=self.markersize_state)
            else:
                ax, handles[i] = self.plot_state(
                        ax=ax, x=pt, color=color,
                        label=txt+str(i), alpha=i/(horizon-1)*4/5+0.2,
                        annotate=annotate,
                        marker=marker_traj,
                        markersize_state=self.markersize_state)

        xlim_old = ax.get_xbound()
        ylim_old = ax.get_ybound()
        max_x = max(np.max(traj[:, 0]), xlim_old[1])
        min_x = min(np.min(traj[:, 0]), xlim_old[0])
        max_y = max(np.max(traj[:, 1]), ylim_old[1])
        min_y = min(np.min(traj[:, 1]), ylim_old[0])
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        return ax, handles

    def plot_ellipsoid_traj(self, pHs_1_T, qHs_1_T, ax=None, plot_lines=False):
        """
        Plot a trajectory of ellipsoids for the human reachability.

        Parameters
        ----------
        pHs_1_T: centers of ellipsoids from time step 1 to T.
        qHs_1_T: shape matrices of ellipsoids from time step 1 to T.
        ax: Axes Object
            The axes to plot the state on
        Returns
        -------
        ax: Axes Object
            The axes with the trajectory plotted
        """
        # new_ax = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # new_ax = True
        plt.sca(ax)

        horizon = pHs_1_T.shape[0]
        assert horizon == qHs_1_T.shape[0]
        assert pHs_1_T.shape[1] == self.n_pH
        assert qHs_1_T.shape[1] == self.n_pH**2

        handles = []
        # https://stackoverflow.com/a/25408562
        color_range = np.linspace(0., 1., num=horizon, endpoint=True)

        for i in range(horizon):
            color = self.ellipsoid_cmap(color_range[i])
            p_i = cas_reshape(pHs_1_T[i, :], (self.n_pH, 1))
            q_i = cas_reshape(qHs_1_T[i, :], (self.n_pH, self.n_pH))
            ax_, handle_ = plot_ellipsoid_2D(
                    centroid=p_i.toarray(), Q=q_i.toarray(),
                    ax=ax, color=color)
            # Ellipsoid computation encounters nan
            if ax_ is None:
                assert handle_ is None
            else:
                assert handle_ is not None
                handles.append(handle_)
                ax = ax_
        if plot_lines:
            p_is = []
            for i in range(horizon):
                p_i = cas_reshape(pHs_1_T[i, :], (self.n_pH, 1))
                p_is.append(p_i.toarray())
            for i in range(horizon-1):
                color = self.ellipsoid_cmap(color_range[i])
                handle = ax.plot(
                        [p_is[i][0], p_is[i+1][0]],
                        [p_is[i][1], p_is[i+1][1]], color=color)
            handles.extend([handle])

        # Create legend
        legend_handles = []
        for i in range(horizon):
            color = self.ellipsoid_cmap(color_range[i])
            patch = mpatches.Patch(color=color, label=str(i))
            legend_handles.append(patch)
        plt.legend(handles=legend_handles)
        handles.extend(legend_handles)

        return ax, handles

    def check_safety_interp(self, cur_pR, cur_vR, cur_pH, cur_vH,
                            next_pR, next_vR, next_pH, next_vH):
        """
        Check safety by conducting cubic interpolation
        between cur_xx and next_xx, and checking safety at every dt.

        Parameters
        ----------
        cur_pR: current robot position.
        cur_vR: current robot velocity.
        cur_pH: current human position.
        cur_vH: current human velocity.
        next_pR: next robot position.
        next_vR: next robot velocity.
        next_pH: next human position.
        next_vH: next human velocity.

        Returns
        -------
        collision: bool, whether human and robot are in collision
                   within this interpolated trajectory.
        safe_impact: bool, whether human and robot have safe impact
                     within this interpolated trajectory.
        min_HR_min_dists: float, min separation distance between human
                          and robot within this interpolated trajectory.
        max_HR_vel_diff: float, max velocity difference between human
                         and robot within this interpolated trajectory.
        """

        assert self.n_pH == self.n_pR

        dts_low_res = [0.0, self.step_time]
        horizon = int(np.ceil(
            self.step_time / self.dt_pH_pR_safety_checking)) + 1
        dts_high_res, step = np.linspace(
                start=0., stop=self.step_time,
                num=horizon, endpoint=True, retstep=True)
        assert step <= self.dt_pH_pR_safety_checking

        # pR, vR
        y = np.vstack((cur_pR.squeeze(), next_pR.squeeze()))
        dydt = np.vstack((cur_vR.squeeze(), next_vR.squeeze()))
        pR_spline = interpolate.CubicHermiteSpline(
                x=dts_low_res, y=y, dydx=dydt, extrapolate=False)
        vR_spline = pR_spline.derivative()

        # pH, vH
        y = np.vstack((cur_pH.squeeze(), next_pH.squeeze()))
        dydt = np.vstack((cur_vH.squeeze(), next_vH.squeeze()))
        pH_spline = interpolate.CubicHermiteSpline(
                x=dts_low_res, y=y, dydx=dydt, extrapolate=False)
        vH_spline = pH_spline.derivative()

        pR_interp = pR_spline(dts_high_res)
        vR_interp = vR_spline(dts_high_res)
        pH_interp = pH_spline(dts_high_res)
        vH_interp = vH_spline(dts_high_res)

        collision = False
        safe_impact = True
        HR_min_dists = []
        HR_vel_diffs = []
        for i in range(len(dts_high_res)):
            pR = pR_interp[i, :].reshape((self.n_pR, 1))
            vR = vR_interp[i, :].reshape((self.n_vR, 1))
            pH = pH_interp[i, :].reshape((self.n_pH, 1))
            vH = vH_interp[i, :].reshape((self.n_vH, 1))
            collisions_, collision_dists_, vel_diff_, safe_impact_\
                = self.check_safety_single_state(pR=pR, vR=vR, pH=pH, vH=vH)
            if True in collisions_:
                collision = True
            if not safe_impact_:
                safe_impact = False
            HR_min_dists.append(min(collision_dists_))
            HR_vel_diffs.append(vel_diff_)
        min_HR_min_dists = float(min(HR_min_dists))
        max_HR_vel_diff = float(max(HR_vel_diffs))
        if collision:
            assert min_HR_min_dists < self.pH_pR_min_sep_dist
        return collision, safe_impact, min_HR_min_dists, max_HR_vel_diff

    def check_safety_single_state(self, pR, pH, vR=None, vH=None):
        """
        Check safety for a single state.

        Parameters
        ----------
        pR: current robot position.
        pH: current human position.
        vR: current robot velocity.
        vH: current human velocity.

        Returns
        -------
        collision: list of bools, collision or not for each point to check.
        collision_dists: list of floats, seperation distance for each point to check.
        norm_v_diff: float, velocity difference between human and robot.
        safe_impact: bool, whether human and robot have safe impact.
        """

        assert pR.shape == (self.n_pR, 1)
        if vR is not None:
            assert vR.shape == (self.n_vR, 1)
        assert pH.shape == (self.n_pH, 1)
        if vH is not None:
            assert vH.shape == (self.n_vH, 1)

        # Collision avoidance
        n_pts = self.R_col_volume_offsets.shape[0]
        collisions = []
        collision_dists = []
        for j in range(n_pts):
            offset = self.R_col_volume_offsets[j, :]
            offset = np.reshape(offset, (self.n_pR, 1))
            assert pR.shape == offset.shape
            coll_pt = pR + offset
            d = np.linalg.norm(coll_pt - pH, ord=2)
            collision_dists.append(d)
            # Since we have already used R_col_volume_offsets,
            # we don't need to add R volume here.
            collisions.append(d < self.pH_pR_min_sep_dist)

        # Safe impact
        # Cst: h_mat x vH <= [-vR, vR]^T + h
        # Here we don't use the over-approximated cst:
        # h_mat x vH <= [-vR, vR]^T + h.
        # Instead, we check safe impact exactly:
        # l*||v1j-v2j||_2 <= F_max
        norm_v_diff = np.linalg.norm(vR - vH, ord=2)
        tmp = abs((self.coeff_restit + 1.) / (1./self.mR + 1./self.mH))

        LFS = norm_v_diff * tmp
        RHS = self.F_HR_max_safe_impact
        safe_impact = LFS <= RHS
        return collisions, collision_dists, norm_v_diff, safe_impact


if __name__ == "__main__":
    # Test
    np.random.seed(0)
    # https://stackoverflow.com/a/2891805
    np.set_printoptions(precision=3, suppress=True)

    config_dir_name = "config_2d_simple"

    path = os.path.abspath(hr_planning.__file__)
    module_dir = os.path.split(path)[0]
    config_dir = os.path.join(
            module_dir,
            "env_gridworld_human/" + config_dir_name)
    config_path_hr_env = os.path.join(config_dir, "hr_env.yaml")
    config_path_hmdp = os.path.join(config_dir, "hmdp.yaml")
    cache_dir = os.path.join(module_dir, "env_gridworld_human/cache")

    # for pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]:
    # for pH_mode in ["pH_indep_pR", "pH_avoid_pR"]:
    for pH_mode in ["pH_avoid_pR"]:
        env = HREnv(config_path_hr_env=config_path_hr_env,
                    config_path_hmdp=config_path_hmdp,
                    cache_dir=cache_dir,
                    pH_mode=pH_mode,
                    value_iteration=True)
        _, ax = env.plot_safety_bounds(plot_human_grid=True, ax=None)
        # plt.show()

        mean = np.array([1, 2, 3, 4])
        std = 0.1
        n_samples = 2
        sampled_sR = env.sample_init_sR(mean=mean, std=std, n_samples=n_samples)
        assert sampled_sR.shape == (n_samples, mean.shape[0])

        pR_0 = np.array([0.3, 0.0]).reshape((env.n_pR, 1))
        vR_0 = np.array([0.0, 0.0]).reshape((env.n_vR, 1))
        pH_0 = np.array([0.1, 0.0]).reshape((env.n_pH, 1))
        vH_0 = np.array([0.0, 0.0]).reshape((env.n_vH, 1))
        cur_pR, cur_vR, cur_pH, cur_vH = env.reset(
                pR_0=pR_0, vR_0=vR_0, pH_0=pH_0, vH_0=vH_0)

        horizon = 10
        pRs_0_T = np.zeros((horizon+1, env.n_pR), dtype=np.float32)
        vRs_0_T = np.zeros((horizon+1, env.n_vR), dtype=np.float32)
        pHs_0_T = np.zeros((horizon+1, env.n_pH), dtype=np.float32)
        vHs_0_T = np.zeros((horizon+1, env.n_vH), dtype=np.float32)
        pRs_0_T[0, :] = np.squeeze(env.cur_pR)
        vRs_0_T[0, :] = np.squeeze(env.cur_vR)
        pHs_0_T[0, :] = np.squeeze(env.cur_pH)
        vHs_0_T[0, :] = np.squeeze(env.cur_vH)

        # We can verify using vt=v0+a*t, xt=x0+v0*t+0.5*a*t^2
        for t in range(1, horizon + 1):
            print_OK("----\nItr={}".format(t))
            # uR = env.random_uR()
            # uRs_1_T = np.array([[0.1, 0.1]])
            uRs_1_T = np.array([[0.0, 0.0]])
            # uRs_1_T = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            assert uRs_1_T.shape[1] == env.n_uR
            next_pH, next_vH, next_pR, next_vR,\
                collision, safe_impact, HR_min_dist, HR_max_vel_diff\
                = env.step(uRs_1_T=uRs_1_T, set_cur_state=True)
            pRs_0_T[t, :] = np.squeeze(env.cur_pR)
            vRs_0_T[t, :] = np.squeeze(env.cur_vR)
            pHs_0_T[t, :] = np.squeeze(env.cur_pH)
            vHs_0_T[t, :] = np.squeeze(env.cur_vH)

        print("pRs_0_T={}".format(pRs_0_T))
        print("pHs_0_T={}".format(pHs_0_T))
        ax, handles = env.plot_traj(traj=pHs_0_T, human_or_robot="H", ax=ax)
        ax, handles = env.plot_traj(traj=pRs_0_T, human_or_robot="R", ax=ax)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
    print("Done")
