# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import numpy as np
from IPython import embed
import yaml
import os
import statistics
import pandas as pd
from scipy import stats
import copy
import math

import hr_planning
from hr_planning.visualization.utils_visualization import print_FAIL, print_OK


# Only log the first itr => fair comparison with the same training data.
first_exp_only = True

# Task variations
pR_modes = ["CA", "CA_SI", "greedy"]
pH_modes = ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]
tasks = ["handover", "coll_avoid"]

info_gain_stop = 8
info_gain_num = 3
info_gain_step = math.ceil(info_gain_stop / info_gain_num)
info_gain_itrs = np.linspace(
        start=0, stop=info_gain_stop, num=info_gain_num,
        endpoint=True, dtype=int).tolist()
info_gain_itrs = []

keys = ["mean_HR_max_vel_diff_all_itrs",
        "mean_HR_min_dist_all_itrs",
        "mean_planning_time_all_itrs",
        "num_unsafe", "perc_unsafe",
        "num_coll", "perc_coll"]

keys_for_csv = ["mean_HR_max_vel_diff_all_itrs",
                "mean_HR_min_dist_all_itrs",
                "mean_planning_time_all_itrs",
                "num_unsafe", "perc_unsafe",
                "num_coll", "perc_coll",
                "n_itrs_reach_goal", "n_feas_ratio",
                "perc_failed_reach_goal"]
for ig_i in info_gain_itrs:
    keys_for_csv.append("IG_itr_" + str(ig_i))

key_2_shorthand = {"mean_HR_max_vel_diff_all_itrs": "mean_maxVelDiff",
                   "mean_HR_min_dist_all_itrs": "mean_minSepDist",
                   "mean_planning_time_all_itrs": "mean_planTime"}


def main():
    path = os.path.abspath(hr_planning.__file__)
    module_dir = os.path.split(path)[0]
    cache_dir = os.path.join(module_dir, "env_gridworld_human/cache")
    map_dirs = sorted([f.path for f in os.scandir(cache_dir) if f.is_dir()])

    # We have multiple maps
    for map_dir in map_dirs:
        dirs = sorted([f.path for f in os.scandir(map_dir) if f.is_dir()])
        all_res = {}
        for exp_dir in dirs:
            exp_name = exp_dir[exp_dir.rfind("/") + 1:]
            print_OK("exp_name={}".format(exp_name))

            exp_config_path = os.path.join(exp_dir, "exp_config.yaml")
            exp_config = None
            with open(exp_config_path) as f:
                exp_config = yaml.load(f, Loader=yaml.FullLoader)
            pR_mode = exp_config["pR_mode"]
            task = exp_config["task"]
            pH_mode = exp_config["pH_mode"]
            seed = exp_config["seed"]

            rmpc_config_path = os.path.join(exp_dir, "rmpc.yaml")
            rmpc_config = None
            with open(rmpc_config_path) as f:
                rmpc_config = yaml.load(f, Loader=yaml.FullLoader)
            n_exps = rmpc_config["n_experiments"]
            n_itrs = rmpc_config["n_iterations"]

            combined = {}
            combined["pR_mode"] = pR_mode
            combined["pH_mode"] = pH_mode
            combined["task"] = task
            combined["exp_dir"] = exp_dir
            for k in keys:
                combined[k] = {}
                combined[k]["data"] = []
            combined["n_itrs_reach_goal"] = {}
            combined["n_itrs_reach_goal"]["data"] = []
            combined["n_feas_ratio"] = {}
            combined["n_feas_ratio"]["data"] = []
            combined["count_failed_reach_goal"] = 0

            for ig_i in info_gain_itrs:
                key = "IG_itr_" + str(ig_i)
                if key not in combined:
                    combined[key] = {}
                    combined[key]["data"] = []
                if ig_i >= len(exp_config["info_gain_all_exps_flat"]):
                    print_FAIL("info_gain_all_exps_flat is too short!")
                    print(len(exp_config["info_gain_all_exps_flat"]))
                    embed()
                tmp = exp_config["info_gain_all_exps_flat"][ig_i]
                # Total info gain
                combined[key]["data"].append(sum(tmp))

            subdirs = sorted([x[0] for x in os.walk(exp_dir)][1:])
            # Ensure ordering is correct:
            indices = []
            for d in subdirs:
                tmp = d[d.rfind("_")+1:]
                idx = int(tmp)
                indices.append(idx)
            assert list(indices) == list(range(len(indices)))
            assert n_exps == len(subdirs)

            for d in subdirs:
                if first_exp_only:
                    if d[d.rfind("/")+1:] != "exp_00":
                        continue
                yaml_path = os.path.join(d, "results.yaml")
                results = None
                with open(yaml_path) as f:
                    results = yaml.load(f, Loader=yaml.FullLoader)

                if "summary" not in results:
                    print("No summmary in {}".format(d))
                    embed()
                summary = results["summary"]

                if results["goal_reaching_at_the_beginning_of_itr"] == -1:
                    combined["count_failed_reach_goal"] += 1
                    combined["n_itrs_reach_goal"]["data"].append(n_itrs)
                else:
                    combined["n_itrs_reach_goal"]["data"].append(
                            results["goal_reaching_at_the_beginning_of_itr"])
                if False in results["init_safe_all_itrs"]:
                    print("NOT init_safe={}".format(d))

                tmp = results["feasibility_all_itrs"]
                r = tmp.count("feasible_plot_new_path") / len(tmp)
                combined["n_feas_ratio"]["data"].append(r)

                for k in keys:
                    pt = summary[k]
                    if type(summary[k]) is list:
                        assert len(summary[k]) == 1
                        pt = summary[k][0]
                    # print("{}={}".format(k, pt))
                    assert type(pt) in [float, int]
                    combined[k]["data"].append(pt)

            tmp = combined["count_failed_reach_goal"] / n_exps
            combined["perc_failed_reach_goal"] = {}
            combined["perc_failed_reach_goal"]["data"] = [tmp]

            # If never reach goal
            if len(combined["n_itrs_reach_goal"]["data"]) <= 0:
                combined["n_itrs_reach_goal"]["data"] = [-1]

            all_keys = copy.deepcopy(keys)
            all_keys += ["n_itrs_reach_goal", "n_feas_ratio",
                         "perc_failed_reach_goal"]
            for ig_i in info_gain_itrs:
                all_keys.append("IG_itr_" + str(ig_i))
            for k in all_keys:
                data = combined[k]["data"]
                assert len(data) > 0
                combined[k]["count"] = len(data)
                combined[k]["mean"] = statistics.mean(data)
                combined[k]["median"] = statistics.median(data)
                combined[k]["max"] = max(data)
                combined[k]["min"] = min(data)
                if len(data) > 1:
                    combined[k]["stdev"] = float(statistics.stdev(data))
                    combined[k]["stderr"] = float(stats.sem(data))

            yaml_path = os.path.join(exp_dir, "combined.yaml")
            with open(yaml_path, 'w') as f:
                yaml.dump(combined, f)

            for k in keys_for_csv:
                if task not in all_res:
                    all_res[task] = {}
                if pR_mode not in all_res[task]:
                    all_res[task][pR_mode] = {}
                if pH_mode not in all_res[task][pR_mode]:
                    all_res[task][pR_mode][pH_mode] = {}
                if k not in all_res[task][pR_mode][pH_mode]:
                    all_res[task][pR_mode][pH_mode][k] = {}
                    all_res[task][pR_mode][pH_mode][k]["data"] = []
                    all_res[task][pR_mode][pH_mode]["seed"] = []
                all_res[task][pR_mode][pH_mode][k]["data"].append(
                        combined[k]["mean"])
            all_res[task][pR_mode][pH_mode]["seed"].append(seed)

        for task in tasks:
            if task not in all_res:
                continue
            for pR_mode in pR_modes:
                if pR_mode not in all_res[task]:
                    continue
                for pH_mode in pH_modes:
                    if pH_mode not in all_res[task][pR_mode]:
                        continue
                    for k in keys_for_csv:
                        data = all_res[task][pR_mode][pH_mode][k]["data"]
                        all_res[task][pR_mode][pH_mode][k]["count"] = len(data)
                        all_res[task][pR_mode][pH_mode][k]["mean"] = statistics.mean(data)
                        all_res[task][pR_mode][pH_mode][k]["max"] = max(data)
                        all_res[task][pR_mode][pH_mode][k]["min"] = min(data)
                        # If we only have 1 data pt, no stdev.
                        if len(data) > 1:
                            stdev = float(statistics.stdev(data))
                            all_res[task][pR_mode][pH_mode][k]["stdev"] = stdev
                            stderr = float(stats.sem(data))
                            all_res[task][pR_mode][pH_mode][k]["stderr"] = stderr
                            if stdev > 1e-5:
                                assert stdev > stderr

        yaml_path = os.path.join(map_dir, "all_res.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(all_res, f)

        entries = ["count", "mean", "stderr"]
        table_column_names = []
        for k in keys_for_csv:
            for c in entries:
                key_str = k
                if k in key_2_shorthand:
                    key_str = key_2_shorthand[k]
                table_column_names.append(key_str + "@" + c)
        table_column_names = sorted(table_column_names)
        print(table_column_names)

        table_indices = []
        for pH_mode in pH_modes:
            for task in tasks:
                for pR_mode in pR_modes:
                    if task not in all_res:
                        continue
                    if pR_mode not in all_res[task]:
                        continue
                    if pH_mode not in all_res[task][pR_mode]:
                        continue
                    table_indices.append(task + "@" + pR_mode + "@" + pH_mode)
        print(table_indices)

        # Export
        # https://github.com/pidipidi/NLP_TAMP/blob/aug_debug2/gazebo_evaluation/src/gazebo_evaluation/data_process_use_last_seed.py
        df = pd.DataFrame(columns=table_column_names, index=table_indices)
        for task in tasks:
            if task not in all_res:
                continue
            for pR_mode in pR_modes:
                if pR_mode not in all_res[task]:
                    continue
                for pH_mode in pH_modes:
                    if pH_mode not in all_res[task][pR_mode]:
                        continue
                    table_idx = task + "@" + pR_mode + "@" + pH_mode
                    for k in keys_for_csv:
                        key_str = k
                        if k in key_2_shorthand:
                            key_str = key_2_shorthand[k]
                        for c in entries:
                            column_name = key_str + "@" + c
                            # If we only have 1 data pt, no stdev nor stderr.
                            entry = all_res[task][pR_mode][pH_mode][k]
                            if c in entry:
                                df.loc[table_idx][column_name] = entry[c]
        path = os.path.join(map_dir, "all_res.csv")
        print(path)
        df.to_csv(path)


if __name__ == "__main__":
    main()
