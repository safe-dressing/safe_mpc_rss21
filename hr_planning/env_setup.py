# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import numpy as np
from datetime import datetime
from IPython import embed
import os
import yaml
import matplotlib.pyplot as plt

import hr_planning
from hr_planning.env_gridworld_human.env import HREnv
from hr_planning.safempc_simple import SimpleSafeMPC
from hr_planning.ssm_gpy.gaussian_process import SimpleGPModel
from hr_planning.visualization.utils_visualization import print_OK


class EnvSetup():
    def __init__(self, env, pH_mode, config_path_rmpc, result_dir, plot_data=False):
        """
        A wrapper class of the human environment,
        GP model (trained with initial dataset), initialized MPC planner.

        Parameters
        ----------
        env: HREnv class
        pH_mode: str
                 pH_indep_pR: H-Indep-R condition in the paper.
                 pH_avoid_pR: H-Away-R condition in the paper.
                 pH_move_to_pR: H-To-R condition in the paper.
        config_path_rmpc: path to the configuration file of the robot MPC.
        result_dir: path to save results.
        plot_data: whether to plot the training data of GP or not.
        """

        assert pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]
        self.pH_mode = pH_mode
        self.env = env
        with open(config_path_rmpc) as f:
            self.config_r_mpc = yaml.load(f, Loader=yaml.FullLoader)

        # XXX: dt_Rmpc must = step time, because we collect human data
        # in the freq of step_time
        # => pH prediction acts in a freq of step_time
        # => robot planning has to act at a freq of step_time.
        assert self.env.step_time == self.env.dt_Rmpc,\
            "dt_Rmpc must = step_time"

        self.n_iterations = self.config_r_mpc["n_iterations"]
        self.n_experiments = self.config_r_mpc["n_experiments"]
        self.visualize = self.config_r_mpc["visualize"]
        self.save_vis = self.config_r_mpc["save_vis"]
        self.retrain_gp_during_1_itr = self.config_r_mpc["retrain_gp_during_1_itr"]
        self.pH_rollout_max_length = self.config_r_mpc["pH_rollout_max_length"]

        # Subset of data of size m for training
        m_gp = self.config_r_mpc["m"]
        kern_types = self.config_r_mpc["kern_types"]
        assert len(kern_types) == self.env.n_pH
        # Inducing points
        Z = None

        kernel_hyp = None
        if "kernel_hyp" in self.config_r_mpc:
            kernel_hyp = self.config_r_mpc["kernel_hyp"]
        self.gp = SimpleGPModel(n_s_out=self.env.n_pH,
                                n_s_in=self.env.n_pH,
                                n_u=self.env.n_pR, m=m_gp,
                                kern_types=kern_types, Z=Z,
                                hyp=kernel_hyp)
        # GP hyp config
        # Code is based on: https://github.com/befelix/safe-exploration/blob/master/experiments/journal_experiment_configs/defaultconfig_exploration.py
        if "kernel_hyp" in self.config_r_mpc:
            for i, h in enumerate(self.gp.hyp):
                for k, v in h.items():
                    assert (kernel_hyp[i][k] == v).all()
            print_OK("Use predefined hyperparam")

        self.safempc = SimpleSafeMPC(env=self.env, ssm=self.gp,
                                     config_path_rmpc=config_path_rmpc,
                                     result_dir=result_dir)

        # -------------------------
        # Use safe policy to rollout until collect enough sample pairs
        # X=m x (npH+npH), y=m x nvH, e.g. m=25
        # Input to the GP
        X = np.zeros((1, self.env.n_pH + self.env.n_pR))
        # Output from the GP
        y = np.zeros((1, self.env.n_pH))

        n_success = 0
        cont = True
        pH_subs_init_data = self.config_r_mpc["pH_subs_init_data"]
        for init_pH_ind in pH_subs_init_data:
            pH_0 = self.env.Hmdp.ss.sub2Positions(tuple(init_pH_ind))
            pH_0 = np.reshape(np.array(pH_0), (self.env.n_pH, 1))
            cur_pR, cur_vR, cur_pH, cur_vH = self.env.reset(
                    pR_0=None, vR_0=None, pH_0=pH_0, vH_0=None)
            rollout_length = 0
            while True:
                rollout_length += 1
                print_OK("----\nItr={}".format(n_success))
                uR = self.safempc.safe_policy(
                        pR=cur_pR, vR=cur_vR, pH=cur_pH, vH=cur_vH)
                uRs_1_T = np.reshape(uR, (1, self.env.n_uR))
                # uRs_1_T = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
                assert uRs_1_T.shape[1] == env.n_uR
                next_pH, next_vH, next_pR, next_vR,\
                    collision, safe_impact, HR_min_dist, HR_max_vel_diff\
                    = self.env.step(
                        uRs_1_T=uRs_1_T, cur_pR=cur_pR, cur_vR=cur_vR,
                        cur_pH=cur_pH, cur_vH=cur_vH,
                        set_cur_state=False)

                inputs_to_gp = np.hstack((cur_pH.squeeze(), cur_pR.squeeze()))
                X = np.vstack((X, inputs_to_gp))
                # XXX: next_pH = cur_pH + gp
                outputs_from_gp = next_pH - cur_pH
                y = np.vstack((y, outputs_from_gp.squeeze()))

                n_success += 1
                '''
                if m_gp is not None:
                    if n_success >= m_gp:
                        cont = False
                        break
                '''
                next_H_ind = self.env.Hmdp.ss.positions2Ind(
                        tuple(np.squeeze(next_pH)))

                if pH_mode in ["pH_indep_pR", "pH_avoid_pR"]:
                    if next_H_ind == self.env.Hmdp.ind_goal:
                        break

                if rollout_length > self.pH_rollout_max_length:
                    break

                cur_pR = next_pR
                cur_vR = next_vR
                cur_pH = next_pH
                cur_vH = next_vH

            if not cont:
                break

        X = X[1:, :]
        y = y[1:, :]

        # plot the training set X
        if plot_data:
            fig, ax = self.env.plot_safety_bounds(
                    plot_human_grid=True, ax=None)
            X_pH = X[:, :self.env.n_pH]
            X_pR = X[:, self.env.n_pH:]
            ax, handles = env.plot_traj(traj=X_pH, human_or_robot="H", ax=ax)
            ax, handles = env.plot_traj(traj=X_pR, human_or_robot="R", ax=ax)
            plt.show()

        # opt_hyp=True: optimize hyperparameter.
        # replace_old=True: since this is the initial dataset.
        # reinitialize_solver=False: we don't initialize the solver here. We will do it later.
        self.safempc.update_model(
                x=X, y=y,
                opt_hyp=True,
                replace_old=True, reinitialize_solver=False)


if __name__ == "__main__":
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

    current_time = datetime.now().strftime("%H:%M:%S")
    result_dir = os.path.join(
            module_dir, "env_gridworld_human/cache", "exp_" + current_time)
    os.mkdir(result_dir)

    pH_mode = "pH_indep_pR"
    # pH_mode = "pH_avoid_pR"
    # pH_mode = "pH_move_to_pR"
    env = HREnv(config_path_hr_env=config_path_hr_env,
                config_path_hmdp=config_path_hmdp,
                cache_dir=result_dir,
                pH_mode=pH_mode,
                value_iteration=True)

    config_path_rmpc = os.path.join(config_dir, "rmpc.yaml")
    e = EnvSetup(env=env, pH_mode=pH_mode,
                 config_path_rmpc=config_path_rmpc,
                 result_dir=result_dir, plot_data=True)
    print("Done")
