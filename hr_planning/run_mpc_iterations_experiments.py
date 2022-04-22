'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import numpy as np
from datetime import datetime

from IPython import embed
import yaml
import os
import matplotlib.pyplot as plt
import optparse
import statistics

import hr_planning
from hr_planning.env_gridworld_human.env import HREnv
from hr_planning.visualization.utils_visualization import print_FAIL, print_OK
from hr_planning.env_setup import EnvSetup
from hr_planning.utils_ellipsoid import distance_to_center
from hr_planning.utils_cache import copy_yaml

# https://stackoverflow.com/questions/51272814/python-yaml-dumping-pointer-references
yaml.Dumper.ignore_aliases = lambda *args: True

# For plotting in paper
produce_plot = True
# produce_plot = False


def main(seed, pR_mode, task, pH_mode, hmdp_name):
    # 2 algorithms for benchmarking (paper Sec.VI(A)):
    # CA: safety = collision avoidance
    # CA_SI: safety = collision avoidance or safe impact
    assert pR_mode in ["CA", "CA_SI"]
    # Simulated task for testing 2 algorithms (paper Sec.VI(A)):
    # - Only coll_avoid (human-robot collision avoidance) is used in the paper,
    #   to show safe+efficient.
    # - handover (human-robot handover): the robot is trying to reach the human.
    # - dressing_2d: a 2D version of the dressing task (paper Sec.VI(B))
    #   where the robot is moving to the human's shoulder,
    #   while ensuring safety regarding
    #   the approximated human arm as described in Fig.5(a).
    assert task in ["coll_avoid", "handover", "dressing_2d"]
    # 3 types of human behavior modes for benchmarking (paper Sec.VI(A)):
    # pH_indep_pR: H-Indep-R condition in the paper.
    # pH_avoid_pR: H-Away-R condition in the paper.
    # pH_move_to_pR: H-To-R condition in the paper.
    assert pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]

    config_dir_name = "config_2d_simple"
    if task == "dressing_2d":
        config_dir_name = "config_2d_dressing"

    path = os.path.abspath(hr_planning.__file__)
    module_dir = os.path.split(path)[0]
    config_dir = os.path.join(
            module_dir,
            "env_gridworld_human/" + config_dir_name)
    config_path_hr_env = os.path.join(config_dir, "hr_env.yaml")
    config_path_hmdp = os.path.join(config_dir, hmdp_name)
    hmdp_folder_name = hmdp_name[:hmdp_name.find(".")]

    hmdp_dir = os.path.join(
            module_dir, "env_gridworld_human/cache",
            hmdp_folder_name)

    if not os.path.isdir(hmdp_dir):
        os.mkdir(hmdp_dir)

    current_time = datetime.now().strftime("%H:%M:%S")
    exp_dir = os.path.join(
            hmdp_dir, "exp_" + pR_mode + "_" + task + "_"
            + pH_mode + "_" + current_time + "_seed_" + str(seed))
    os.mkdir(exp_dir)

    with open(config_path_hr_env) as f:
        config_hr_env = yaml.load(f, Loader=yaml.FullLoader)
    pR_0_task = None
    if task == "coll_avoid":
        pR_0_task = config_hr_env["pR_0_coll_avoid"]
    elif task == "handover":
        pR_0_task = config_hr_env["pR_0_handover"]
    elif task == "dressing_2d":
        pR_0_task = config_hr_env["pR_0_dressing_2d"]
    else:
        raise ValueError()

    env = HREnv(config_path_hr_env=config_path_hr_env,
                config_path_hmdp=config_path_hmdp,
                cache_dir=exp_dir, pH_mode=pH_mode,
                value_iteration=True, pR_0_arg=pR_0_task)

    config_path_rmpc = os.path.join(config_dir, "rmpc.yaml")
    cfg = EnvSetup(env=env, pH_mode=pH_mode,
                   config_path_rmpc=config_path_rmpc,
                   result_dir=exp_dir, plot_data=False)

    path = os.path.join(exp_dir, "hr_env.yaml")
    copy_yaml(in_path=config_path_hr_env, out_path=path)

    path = os.path.join(exp_dir, "hmdp.yaml")
    copy_yaml(in_path=config_path_hmdp, out_path=path)

    path = os.path.join(exp_dir, "rmpc.yaml")
    copy_yaml(in_path=config_path_rmpc, out_path=path)

    exp_logs = {}
    exp_logs["seed"] = seed
    exp_logs["pR_mode"] = pR_mode
    exp_logs["task"] = task
    exp_logs["pH_mode"] = pH_mode
    exp_logs["info_gain"] = {}

    yaml_path = os.path.join(exp_dir, "exp_config.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(exp_logs, f)

    cfg.safempc.set_task(pR_mode=pR_mode, task=task)

    for exp_idx in range(cfg.n_experiments):
        exp_idx_str = str(exp_idx).zfill(2)
        print_FAIL("At exp={}".format(exp_idx_str))
        itrs_dir = os.path.join(exp_dir, "exp_{}".format(exp_idx_str))
        os.mkdir(itrs_dir)
        itr_logs = {}

        cfg.safempc.reset()
        cfg.safempc.result_dir = itrs_dir

        # Need to init solver after updating GP.
        cfg.safempc.init_solver()

        cur_pR, cur_vR, cur_pH, cur_vH = cfg.env.reset(
                pR_0=None, vR_0=None, pH_0=None, vH_0=None)
        prev_pR = np.copy(cur_pR)
        prev_vR = np.copy(cur_vR)
        prev_pH = np.copy(cur_pH)
        prev_vH = np.copy(cur_vH)

        itr_logs["init_safe_all_itrs"] = []
        itr_logs["collision_all_itrs"] = []
        itr_logs["safe_impact_all_itrs"] = []
        itr_logs["HR_min_dist_all_itrs"] = []
        itr_logs["HR_max_vel_diff_all_itrs"] = []
        itr_logs["max_itr"] = -1
        itr_logs["planning_time_all_itrs"] = []
        itr_logs["feasibility_all_itrs"] = []
        itr_logs["goal_reaching_at_the_beginning_of_itr"] = -1
        itr_logs["info_gain_all_itrs"] = []

        for itr_idx in range(cfg.n_iterations):
            itr_logs[itr_idx] = {}

            print_FAIL("At itr={}".format(itr_idx))
            print("pR={}".format(np.squeeze(cur_pR)))
            print("vR={}".format(np.squeeze(cur_vR)))
            print("pH={}, vH={}".format(np.squeeze(cur_pH), np.squeeze(cur_vH)))

            cur_pH_sub = cfg.env.Hmdp.ss.positions2Sub(tuple(cur_pH.squeeze()))
            print("cur_pH_sub={}".format(cur_pH_sub))

            # -------------------------------
            # 1. Check whether the current state is safe or not
            collision, safe_impact, HR_min_dist, HR_max_vel_diff\
                = cfg.env.check_safety_interp(
                        cur_pR=prev_pR, cur_vR=prev_vR,
                        cur_pH=prev_pH, cur_vH=prev_vH,
                        next_pR=cur_pR, next_vR=cur_vR,
                        next_pH=cur_pH, next_vH=cur_vH)
            init_safe = True
            if collision and not safe_impact:
                init_safe = False
            if not init_safe:
                print_FAIL("Not init_safe at itr={}".format(itr_idx))
            else:
                print_OK("init_safe at itr={}".format(itr_idx))

            itr_logs[itr_idx]["init"] = {}
            itr_logs[itr_idx]["init"]["cur_pH"] = np.squeeze(cur_pH).tolist()
            itr_logs[itr_idx]["init"]["cur_vH"] = np.squeeze(cur_vH).tolist()
            itr_logs[itr_idx]["init"]["cur_pR"] = np.squeeze(cur_pR).tolist()
            itr_logs[itr_idx]["init"]["cur_vR"] = np.squeeze(cur_vR).tolist()

            itr_logs[itr_idx]["init"]["collision"] = collision
            itr_logs[itr_idx]["init"]["safe_impact"] = safe_impact
            itr_logs[itr_idx]["init"]["HR_min_dist"] = HR_min_dist
            itr_logs[itr_idx]["init"]["HR_max_vel_diff"] = HR_max_vel_diff
            itr_logs[itr_idx]["init"]["init_safe"] = init_safe

            itr_logs["init_safe_all_itrs"].append(init_safe)
            itr_logs["collision_all_itrs"].append(collision)
            itr_logs["safe_impact_all_itrs"].append(safe_impact)
            itr_logs["HR_min_dist_all_itrs"].append(HR_min_dist)
            itr_logs["HR_max_vel_diff_all_itrs"].append(HR_max_vel_diff)
            itr_logs["max_itr"] = itr_idx

            # -------------------------------
            # 1.5 Check whether the goal is reached or not
            if task in ["coll_avoid", "dressing_2d"]:
                assert cur_pR.shape == cfg.env.pR_goal.shape
                d_2_goal = np.linalg.norm(cur_pR - cfg.env.pR_goal, ord=2)
                if d_2_goal < cfg.env.pR_goal_tol:
                    print_OK("Goal is reached")
                    itr_logs[itr_idx]["goal_is_reached"] = True
                    itr_logs["goal_reaching_at_the_beginning_of_itr"] = itr_idx
                    break
            elif task == "handover":
                # assert cur_pH.shape == cur_pR.shape
                # d_HR = np.linalg.norm(cur_pH - cur_pR, ord=2)
                if HR_min_dist < cfg.env.pR_goal_tol:
                    print_OK("Handover goal is reached")
                    itr_logs[itr_idx]["goal_is_reached"] = True
                    itr_logs["goal_reaching_at_the_beginning_of_itr"] = itr_idx
                    break
            else:
                raise ValueError()

            # -------------------------------
            # 2. Plot the starting configuration
            if cfg.visualize or cfg.save_vis:
                if produce_plot:
                    fig, ax = cfg.env.plot_safety_bounds(
                            plot_human_grid=False, ax=None,
                            plot_world_rectangle=False)
                else:
                    fig, ax = cfg.env.plot_safety_bounds(
                            plot_human_grid=True, ax=None)

                # plot the training set
                x_train_init = cfg.safempc.ssm.x_train
                n_train, _ = np.shape(x_train_init)
                for i in range(n_train):
                    ax, handle = cfg.env.plot_state(
                            ax, x_train_init[i, :cfg.env.n_pH],
                            color=cfg.env.color_hmdp_grid,
                            marker=cfg.env.marker_human_data, alpha=0.1,
                            markersize_state=cfg.env.markersize_state)

                if produce_plot:
                    annotate = False
                else:
                    annotate = True

                # Plot pR initial state
                pR_traj = np.reshape(cur_pR, (1, cfg.env.n_pR))
                ax, handles = cfg.env.plot_traj(pR_traj, "R",
                                                ax=ax, annotate=annotate)
                assert len(handles[0]) == 1
                assert len(handles) == 1

                # Plot sH initial state
                sH_traj = np.reshape(cur_pH, (1, cfg.env.n_pH))
                ax, handles = cfg.env.plot_traj(sH_traj, "H",
                                                ax=ax, annotate=annotate)
                assert len(handles[0]) == 1
                assert len(handles) == 1

                # Plot goal state
                if produce_plot:
                    cfg.env.plot_state(
                            ax=ax, x=cfg.env.pR_goal.squeeze(),
                            color="r", label="", alpha=0.4, annotate=False,
                            marker="*", markersize_state=15)
                    cfg.env.plot_state(
                            ax=ax, x=cfg.env.Hmdp.pos_goal.squeeze(),
                            color="g", label="", alpha=0.4, annotate=False,
                            marker="*", markersize_state=15.)
                # plt.show()

            itr_logs[itr_idx]["mpc"] = {}
            u_i = None
            us_1_T_opt = None
            status_str = None
            planning_time = -1
            if not init_safe:
                u_i = cfg.safempc.safe_policy(cur_pR, cur_vR, cur_pH, cur_vR)
                print_FAIL("NOT init_safe => use safe policy at itr={}, ui={}"
                           .format(itr_idx, u_i))
                itr_logs[itr_idx]["mpc"]["status"] = "NOT init_safe"
                status_str = "not_init_safe_no_plot"
            else:
                # -------------------------------
                # 3. If init_safe, try solving the MPC
                pHs_1_T, qHs_1_T, pH_gp_pred_sigma_1_T,\
                    vHs_1_T, vqHs_1_T,\
                    pRs_1_T_opt, vRs_1_T_opt, us_1_T_opt,\
                    status, planning_time\
                    = cfg.safempc.solve_MPC(
                            pR_0=cur_pR, vR_0=cur_vR,
                            pH_0=cur_pH, itr_idx=itr_idx)

                if status == cfg.safempc.status_infeasible_MPC_use_safe_policy:
                    # So just execute the safe policy.
                    u_i = cfg.safempc.safe_policy(cur_pR, cur_vR, cur_pH, cur_vH)
                    print_FAIL("MPC: use safe policy at itr={}, ui={}"
                               .format(itr_idx, u_i))
                    itr_logs[itr_idx]["mpc"]["status"] = "Infeasible; safe_policy"
                    status_str = "infeasible_safe_policy_plot_infeasible_sol"
                else:
                    assert pHs_1_T.shape == (cfg.safempc.h_safe, cfg.env.n_pH)
                    assert qHs_1_T.shape == (cfg.safempc.h_safe, cfg.env.n_pH**2)
                    if pR_mode == "CA_SI":
                        assert vHs_1_T.shape == (cfg.safempc.h_safe, cfg.env.n_pH)
                        assert vqHs_1_T.shape == (cfg.safempc.h_safe, cfg.env.n_pH**2)
                    else:
                        assert vHs_1_T is None
                        assert vqHs_1_T is None
                    assert pRs_1_T_opt.shape == (cfg.safempc.h_safe, cfg.env.n_pR)
                    assert vRs_1_T_opt.shape == (cfg.safempc.h_safe, cfg.env.n_vR)
                    assert us_1_T_opt.shape == (cfg.safempc.h_safe, cfg.env.n_uR)

                    u_i = np.reshape(us_1_T_opt[0, :], (cfg.env.n_uR,))

                    if status == cfg.safempc.status_infeasible_MPC_use_old_traj:
                        print_FAIL("MPC: use old path at itr={}, ui={}"
                                   .format(itr_idx, u_i))
                        itr_logs[itr_idx]["mpc"]["status"] = "Infeasible; old_path"
                        status_str = "infeasible_old_path_plot_shifted_old_path"
                    elif status == cfg.safempc.status_feasible_MPC:
                        print_OK("MPC: feasible at itr={}, ui={}"
                                 .format(itr_idx, u_i))
                        itr_logs[itr_idx]["mpc"]["status"] = "Feasible"
                        status_str = "feasible_plot_new_path"
                    else:
                        raise ValueError()

                    print("cur_pR={}, cur_vR={}, cur_pH={}".format(
                        np.squeeze(cur_pR), np.squeeze(cur_vR),
                        np.squeeze(cur_pH)))
                    print("pRs_1_T_opt")
                    print(pRs_1_T_opt)
                    print("pHs_1_T")
                    print(pHs_1_T.toarray())
                    print("qHs_1_T")
                    print(qHs_1_T.toarray())
                    print("pH_gp_pred_sigma_1_T")
                    print(pH_gp_pred_sigma_1_T.toarray())
                    if pR_mode == "CA_SI":
                        print("vHs_1_T")
                        print(vHs_1_T.toarray())
                        print("vqHs_1_T")
                        print(vqHs_1_T.toarray())

                if cfg.visualize or cfg.save_vis:
                    if produce_plot:
                        annotate = False
                    else:
                        annotate = True

                    # Draw pR traj
                    pRs_0_T_opt = np.vstack((
                        np.reshape(cur_pR, (cfg.env.n_pR)), pRs_1_T_opt))
                    ax, handles = cfg.env.plot_traj(
                            pRs_0_T_opt, "R", ax=ax, annotate=annotate)

                    # Draw pH traj
                    pHs_0_T = np.vstack((
                        np.reshape(cur_pH, (cfg.env.n_pH)), pHs_1_T))
                    ax, handles = cfg.env.plot_traj(
                            pHs_0_T, "H", ax=ax, annotate=annotate)

                    # Draw pH ellipsoidal traj
                    ax, handles = cfg.env.plot_ellipsoid_traj(
                            pHs_1_T, qHs_1_T, ax=ax, plot_lines=False)

                    if task == "dressing_2d":
                        interp_2_pHs_1_T = cfg.safempc.interp_ellipsoids(
                                pHs_1_T.toarray())
                        for k, pHs_1_T_interp in interp_2_pHs_1_T.items():
                            ax, handles = cfg.env.plot_ellipsoid_traj(
                                    pHs_1_T_interp, qHs_1_T,
                                    ax=ax, plot_lines=False)
                    # plt.show()

            assert u_i is not None
            assert status_str is not None
            itr_logs[itr_idx]["mpc"]["u_i_opt"] = np.squeeze(u_i).tolist()
            if us_1_T_opt is not None:
                itr_logs[itr_idx]["mpc"]["us_1_T_opt"] = us_1_T_opt.tolist()

            itr_logs[itr_idx]["mpc"]["planning_time"] = float(planning_time)
            itr_logs["planning_time_all_itrs"].append(planning_time)
            itr_logs["feasibility_all_itrs"].append(status_str)

            itr_logs[itr_idx]["next"] = {}
            # -------------------------------
            # 4. Update for next iteration
            # XXX: Here we pass the whole traj, the step function in env class
            # will determine the robot state at exactly the next timestep.
            uRs_1_T = us_1_T_opt
            if uRs_1_T is None:
                uRs_1_T = u_i
                uRs_1_T = uRs_1_T.reshape(1, cfg.env.n_uR)
            next_pH, next_vH, next_pR, next_vR,\
                collision, safe_impact, HR_min_dist, HR_max_vel_diff\
                = cfg.env.step(
                    uRs_1_T=uRs_1_T, cur_pR=cur_pR, cur_vR=cur_vR,
                    cur_pH=cur_pH, cur_vH=cur_vH,
                    set_cur_state=False)

            # Compare next_vH and Euler integrated next_vH
            print("next_pR={}".format(np.squeeze(next_pR)))
            print("next_vR={}".format(np.squeeze(next_vR)))
            print("next_pH={}".format(np.squeeze(next_pH)))
            print("next_vH={}".format(np.squeeze(next_vH)))
            next_vH_euler = (next_pH - cur_pH) / cfg.env.step_time
            diff = np.linalg.norm(
                    next_vR.squeeze() - next_vH_euler.squeeze(), ord=2)
            print_FAIL("next_vH_euler={}, diff={}".format(
                np.squeeze(next_vH_euler), diff))
            itr_logs[itr_idx]["next"]["next_pR"] = np.squeeze(next_pR).tolist()
            itr_logs[itr_idx]["next"]["next_vR"] = np.squeeze(next_vR).tolist()
            itr_logs[itr_idx]["next"]["next_pH"] = np.squeeze(next_pH).tolist()
            itr_logs[itr_idx]["next"]["next_vH"] = np.squeeze(next_vH).tolist()
            itr_logs[itr_idx]["next"]["next_vH_euler"] = np.squeeze(next_vH_euler).tolist()
            itr_logs[itr_idx]["next"]["diff_next_vH_vs_next_vH_euler"] = float(diff)

            # Check whether next_pH, next_vH is in ellipsoid
            samples = next_pH.T
            p_center = pHs_1_T[0, :].T
            q_shape = qHs_1_T[0, :].reshape((cfg.env.n_pH, cfg.env.n_pH))
            d = distance_to_center(
                    samples=samples, p_center=p_center, q_shape=q_shape)
            if d < 1.:
                print_OK("next_pH is in ellip")
                itr_logs[itr_idx]["next"]["next_pH_in_ellip"] = True
            else:
                print_FAIL("next_pH is NOT in ellip, d={}".format(d))
                itr_logs[itr_idx]["next"]["next_pH_in_ellip"] = False

            if pR_mode == "CA_SI":
                samples = next_vH.T
                p_center = vHs_1_T[0, :].T
                q_shape = vqHs_1_T[0, :].reshape((cfg.env.n_vH, cfg.env.n_vH))
                d = distance_to_center(
                        samples=samples, p_center=p_center, q_shape=q_shape)
                if d < 1.:
                    print_OK("next_vH is in ellip")
                    itr_logs[itr_idx]["next"]["next_vH_in_ellip"] = True

                else:
                    print_FAIL("next_vH is NOT in ellip, d={}".format(d))
                    itr_logs[itr_idx]["next"]["next_vH_in_ellip"] = False

                samples = next_vH_euler.T
                p_center = vHs_1_T[0, :].T
                q_shape = vqHs_1_T[0, :].reshape((cfg.env.n_vH, cfg.env.n_vH))
                d = distance_to_center(
                        samples=samples, p_center=p_center, q_shape=q_shape)
                if d < 1.:
                    itr_logs[itr_idx]["next"]["next_vH_euler_in_ellip"] = True
                    print_OK("next_vH_euler is in ellip")
                else:
                    itr_logs[itr_idx]["next"]["next_vH_euler_in_ellip"] = False
                    print_FAIL("next_vH_euler is NOT in ellip, d={}".format(d))

            next_safe = True
            if collision and not safe_impact:
                next_safe = False
            if not next_safe:
                print_FAIL("Not next_safe at itr={}".format(itr_idx))
                itr_logs[itr_idx]["next"]["next_safe"] = False
            else:
                print_OK("next_safe at itr={}".format(itr_idx))
                itr_logs[itr_idx]["next"]["next_safe"] = True

            new_data_input = np.vstack((cur_pH, cur_pR)).T
            # XXX: next_pH = cur_pH + gp
            assert next_pH.shape == cur_pH.shape
            new_data_output = np.reshape(next_pH - cur_pH, (1, cfg.env.n_pH))

            prev_pH = np.copy(cur_pH)
            prev_vH = np.copy(cur_vH)
            prev_pR = np.copy(cur_pR)
            prev_vR = np.copy(cur_vR)

            cur_pH = np.copy(next_pH)
            cur_vH = np.copy(next_vH)
            cur_pR = np.copy(next_pR)
            cur_vR = np.copy(next_vR)

            if cfg.visualize or cfg.save_vis:
                if not produce_plot:
                    # Plot pH next state
                    ax, handles = cfg.env.plot_state(
                            ax, next_pH.squeeze(), color="blue",
                            label="H'", alpha=0.5,
                            marker="o", markersize_state=20.)
                    ax, handles = cfg.env.plot_state(
                            ax, next_pR.squeeze(), color="red",
                            label="R'", alpha=0.5,
                            marker="o", markersize_state=20.)

            # Update the GP
            # If train=False, update data in sparse GP
            # via k-means && no update hyperparam.
            # If train=True, update data in sparse GP
            # via k-means && update hyperparam.
            # If replace_old=true, it will use the newly collected 1 data sample
            # to replace the entire GP dataset, which is undesired.
            cfg.safempc.update_model(
                    x=new_data_input, y=new_data_output,
                    opt_hyp=cfg.retrain_gp_during_1_itr,
                    replace_old=False,
                    reinitialize_solver=True)
            inf_gain = cfg.safempc.ssm.information_gain()
            itr_logs[itr_idx]["gp"] = {}
            tmp = [float(x) for x in inf_gain]
            itr_logs[itr_idx]["gp"]["info_gain"] = tmp
            itr_logs["info_gain_all_itrs"].append(tmp)

            # -------------------------------
            # 5. Plotting and logging
            if cfg.visualize or cfg.save_vis:
                if not produce_plot:
                    tmp = "IG=["
                    for i in inf_gain:
                        tmp += "{:.3f}, ".format(i)
                    tmp += "]"
                    xlim = ax.get_xbound()
                    ylim = ax.get_ybound()
                    handles = ax.text(xlim[0], ylim[1], s=tmp, fontsize=12)

            # https://stackoverflow.com/a/9012749
            '''
            if cfg.visualize:
                plt.show()
                embed()
            '''

            fig1 = plt.gcf()
            title = status_str
            current_time = datetime.now().strftime("%H:%M:%S")
            tmp = "itr_" + str(itr_idx) + "_"\
                + current_time + "_" + title
            plot_path = os.path.join(cfg.safempc.result_dir, tmp) + ".pdf"
            itr_logs[itr_idx]["plot_path"] = plot_path

            if cfg.save_vis:
                fig1.savefig(plot_path)
                # Sometimes nan appears in ellipsoid drawing,
                # which will fail here.
                '''
                try:
                    fig1.savefig(plot_path)
                except:
                    print_FAIL("Plot fails, when status={}".format(status))
                    embed()
                '''
            plt.close()

            yaml_path = os.path.join(itrs_dir, "results.yaml")
            itr_logs[itr_idx]["yaml_path"] = yaml_path
            with open(yaml_path, 'w') as f:
                yaml.dump(itr_logs, f)

        itr_logs["summary"] = {}
        itr_logs["summary"]["info_gain_all_itrs"]\
            = itr_logs["info_gain_all_itrs"]
        exp_logs["info_gain"][exp_idx] = itr_logs["info_gain_all_itrs"]

        num_unsafe = itr_logs["init_safe_all_itrs"].count(False)
        perc_unsafe = num_unsafe / len(itr_logs["init_safe_all_itrs"])
        itr_logs["summary"]["num_unsafe"] = num_unsafe
        itr_logs["summary"]["perc_unsafe"] = perc_unsafe

        num_coll = itr_logs["collision_all_itrs"].count(True)
        perc_coll = num_coll / len(itr_logs["collision_all_itrs"])
        itr_logs["summary"]["num_coll"] = num_coll
        itr_logs["summary"]["perc_coll"] = perc_coll

        num_safe_imp = itr_logs["safe_impact_all_itrs"].count(True)
        perc_safe_imp = num_safe_imp / len(itr_logs["safe_impact_all_itrs"])
        itr_logs["summary"]["num_safe_imp"] = num_safe_imp
        itr_logs["summary"]["perc_safe_imp"] = perc_safe_imp

        tmp = itr_logs["HR_min_dist_all_itrs"]
        itr_logs["summary"]["mean_HR_min_dist_all_itrs"] = statistics.mean(tmp)
        itr_logs["summary"]["max_HR_min_dist_all_itrs"] = max(tmp)
        itr_logs["summary"]["min_HR_min_dist_all_itrs"] = min(tmp)
        if len(tmp) > 1:
            itr_logs["summary"]["stdev_HR_min_dist_all_itrs"] = statistics.stdev(tmp)

        tmp = itr_logs["HR_max_vel_diff_all_itrs"]
        itr_logs["summary"]["mean_HR_max_vel_diff_all_itrs"] = statistics.mean(tmp)
        itr_logs["summary"]["max_HR_max_vel_diff_all_itrs"] = max(tmp)
        itr_logs["summary"]["min_HR_max_vel_diff_all_itrs"] = min(tmp)
        if len(tmp) > 1:
            itr_logs["summary"]["stdev_HR_max_vel_diff_all_itrs"] = statistics.stdev(tmp)

        tmp = itr_logs["planning_time_all_itrs"]
        itr_logs["summary"]["mean_planning_time_all_itrs"] = statistics.mean(tmp)
        itr_logs["summary"]["max_planning_time_all_itrs"] = max(tmp)
        itr_logs["summary"]["min_planning_time_all_itrs"] = min(tmp)
        if len(tmp) > 1:
            itr_logs["summary"]["stdev_planning_time_all_itrs"] = statistics.stdev(tmp)

        yaml_path = os.path.join(itrs_dir, "results.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(itr_logs, f)

        # Update GP hyp after one itr
        cfg.safempc.update_model(
                x=cfg.safempc.ssm.x_train,
                y=cfg.safempc.ssm.y_train,
                opt_hyp=True,
                replace_old=True,
                reinitialize_solver=True)

    assert set(exp_logs["info_gain"].keys()) == set(range(cfg.n_experiments))
    exp_logs["info_gain_all_exps_flat"] = []
    for exp_idx in range(cfg.n_experiments):
        tmp = exp_logs["info_gain"][exp_idx]
        exp_logs["info_gain_all_exps_flat"].extend(tmp)
    yaml_path = os.path.join(exp_dir, "exp_config.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(exp_logs, f)


if __name__ == "__main__":
    # https://stackoverflow.com/a/2891805
    np.set_printoptions(precision=5, suppress=True)

    p = optparse.OptionParser()
    p.add_option('--pR_mode', action='store', type=str)
    p.add_option('--task', action='store', type=str)
    p.add_option('--pH_mode', action='store', type=str)
    p.add_option('--seed', action='store', type=int)
    p.add_option('--hmdp_name', action='store', type=str)

    opt, args = p.parse_args()

    pR_mode = opt.pR_mode
    assert pR_mode is not None
    assert pR_mode in ["CA", "CA_SI"]
    task = opt.task
    assert task is not None
    assert task in ["coll_avoid", "handover", "dressing_2d"]
    pH_mode = opt.pH_mode
    assert pH_mode is not None
    assert pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]
    seed = opt.seed
    assert seed is not None
    assert type(seed) is int
    if opt.hmdp_name is None:
        hmdp_name = "hmdp.yaml"
    else:
        hmdp_name = opt.hmdp_name

    print_FAIL("\nseed={}, pR_mode={}, task={}, pH_mode={}, hmdp_name={}"
               .format(seed, pR_mode, task, pH_mode, hmdp_name))

    np.random.seed(seed)
    main(seed=seed, pR_mode=pR_mode,
         task=task, pH_mode=pH_mode, hmdp_name=hmdp_name)
    print("Done")
