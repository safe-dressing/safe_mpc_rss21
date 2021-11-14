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
from scipy import stats
import copy
import math
from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare, ttest_ind

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

import hr_planning
from hr_planning.visualization.utils_visualization import print_FAIL, print_OK

items_to_log = ["n_itrs_reach_goal", "num_coll", "mean_HR_min_dist_all_itrs", "mean_planning_time_all_itrs", "perc_coll"]


def main():
    path = os.path.abspath(hr_planning.__file__)
    module_dir = os.path.split(path)[0]
    cache_dir = os.path.join(module_dir, "env_gridworld_human/cache")
    map_dirs = sorted([f.path for f in os.scandir(cache_dir) if f.is_dir()])

    mapName_pHMode_2_pRMode_2_data = {}
    for map_dir in map_dirs:
        map_name = map_dir[map_dir.rfind("/")+1:]
        path = os.path.join(map_dir, "all_res.yaml")
        with open(path) as f:
            result_map = yaml.load(f, Loader=yaml.FullLoader)

        # We only do coll avoid
        result_map_task = result_map["coll_avoid"]

        for pR_mode in ["CA", "CA_SI"]:
            result_map_task_pRMode = result_map_task[pR_mode]

            for pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]:
                result_map_task_pRMode_pHMode = result_map_task_pRMode[pH_mode]

                key = (map_name, pH_mode)
                if key not in mapName_pHMode_2_pRMode_2_data:
                    mapName_pHMode_2_pRMode_2_data[key] = {}
                kkey = pR_mode
                if kkey not in mapName_pHMode_2_pRMode_2_data[key]:
                    mapName_pHMode_2_pRMode_2_data[key][kkey] = {}

                for x in items_to_log:
                    mapName_pHMode_2_pRMode_2_data[key][kkey][x] = {}
                    tmp = result_map_task_pRMode_pHMode[x]
                    assert tmp["count"] == 30
                    mapName_pHMode_2_pRMode_2_data[key][kkey][x]["mean"] = tmp["mean"]
                    mapName_pHMode_2_pRMode_2_data[key][kkey][x]["stdev"] = tmp["stdev"]
                    mapName_pHMode_2_pRMode_2_data[key][kkey][x]["stderr"] = tmp["stderr"]
                    mapName_pHMode_2_pRMode_2_data[key][kkey][x]["data"] = tmp["data"]
    yaml_path = os.path.join(cache_dir, "combined.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(mapName_pHMode_2_pRMode_2_data, f)

    # Generate latex code
    for item in items_to_log:
        print_OK("==============={}".format(item))
        for pR_mode in ["CA_SI", "CA"]:
            print_OK(pR_mode)
            s = "& "
            cols = []
            for map_name in ["hmdp1", "hmdp2", "hmdp3", "hmdp4", "hmdp5"]:
                for pH_mode in ["pH_indep_pR", "pH_move_to_pR", "pH_avoid_pR"]:
                    key = (map_name, pH_mode)
                    cols.append(key)

                    # HARDCODE: Only support 2 pR_modes
                    if pR_mode == "CA_SI":
                        pR_mode_compare = "CA"
                    elif pR_mode == "CA":
                        pR_mode_compare = "CA_SI"
                    else:
                        raise ValueError()
                    tmp = mapName_pHMode_2_pRMode_2_data[key][pR_mode][item]
                    tmp_compare = mapName_pHMode_2_pRMode_2_data[key][pR_mode_compare][item]

                    # Since we will do ANOVA, we only show the mean in the table.
                    mean = tmp["mean"]
                    mean_compare = tmp_compare["mean"]

                    # HARDCODE
                    bold = False
                    if item == "mean_HR_min_dist_all_itrs":
                        if mean > mean_compare - 1e-5:
                            bold = True
                    elif item in ["n_itrs_reach_goal", "num_coll", "perc_coll",
                                  "mean_planning_time_all_itrs"]:
                        if mean < mean_compare - 1e-5:
                            bold = True
                    else:
                        raise ValueError()
                    if bold:
                        s += "$\\mathbf{" + "{:.3f}".format(mean) + "}$ & "
                    else:
                        s += "${:.3f}$".format(mean) + " & "
            s = s[:-3] + "\\\\"
            print(s)
            print(cols)
        print()

    item_2_stat_p = {}
    for item in items_to_log:
        CA_SI_data = []
        CA_data = []
        for pR_mode in ["CA_SI", "CA"]:
            for map_name in ["hmdp1", "hmdp2", "hmdp3", "hmdp4", "hmdp5"]:
                for pH_mode in ["pH_indep_pR", "pH_move_to_pR", "pH_avoid_pR"]:
                    key = (map_name, pH_mode)
                    tmp = mapName_pHMode_2_pRMode_2_data[key][pR_mode][item]
                    data = tmp["data"]
                    if pR_mode == "CA_SI":
                        CA_SI_data.extend(data)
                    else:
                        CA_data.extend(data)

        if item in ["n_itrs_reach_goal", "num_coll"]:
            stat, p = wilcoxon(CA_SI_data, CA_data)
        elif item in ["mean_HR_min_dist_all_itrs", "mean_planning_time_all_itrs", "perc_coll"]:
            stat, p = wilcoxon(CA_SI_data, CA_data)
        else:
            raise ValueError()
        item_2_stat_p[item] = (stat, p)
    print(item_2_stat_p)


if __name__ == "__main__":
    main()
