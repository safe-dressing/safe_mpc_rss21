# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import numpy as np
import os
import hr_planning
from hr_planning.env_gridworld_human.hmdp import HumanMdp

num_grids_in_obs = 40
num_obss = 5
# pH_0 cannot be in obs
collision_free_subs = [(8, 2)]


def main():
    path = os.path.abspath(hr_planning.__file__)
    module_dir = os.path.split(path)[0]
    config_dir = os.path.join(module_dir, "env_gridworld_human/config")
    config_path = os.path.join(config_dir, "hmdp.yaml")
    cache_dir = os.path.join(module_dir, "env_gridworld_human/cache")

    hmdp = HumanMdp(config_path=config_path, cache_dir=cache_dir)
    dims = hmdp.ss.dof_2_dims
    # area_of_cur_obs = len(hmdp.inds_obstacle)

    lb1s = []
    lb2s = []
    ub1s = []
    ub2s = []

    while len(lb1s) < num_obss:
        lb1 = None
        lb2 = None
        ub1 = None
        ub2 = None
        while True:
            x, y = np.random.choice(list(range(dims[0])), size=2, replace=False, p=None)
            lb1 = min(x, y)
            ub1 = max(x, y)
            x, y = np.random.choice(list(range(dims[1])), size=2, replace=False, p=None)
            lb2 = min(x, y)
            ub2 = max(x, y)
            area = abs(lb1 - ub1) * abs(lb2 - ub2)
            if area == num_grids_in_obs:
                break

        collision_free_subs.append(hmdp.sub_goal)
        bad = False
        for sub in collision_free_subs:
            if sub[0] in range(lb1, ub1) and sub[1] in range(lb2, ub2):
                bad = True
                break
        if bad:
            continue

        hmdp.poss_obstacle = []
        hmdp.inds_obstacle = []
        hmdp.subs_obstacle = []
        for i in range(lb1, ub1):
            for j in range(lb2, ub2):
                hmdp.subs_obstacle.append((i, j))
                ind = hmdp.ss.sub2ind((i, j))
                hmdp.inds_obstacle.append(ind)
                pos = hmdp.ss.sub2Positions((i, j))
                hmdp.poss_obstacle.append(pos)
        assert len(hmdp.inds_obstacle) == num_grids_in_obs
        hmdp.computeTransitionAndRewardArrays()
        hmdp.value_iteration(discount=1, epsilon=0.01, max_iter=1000)
        hmdp.printPolicy()

        lb1s.append(lb1)
        ub1s.append(ub1-1)
        lb2s.append(lb2)
        ub2s.append(ub2-1)

    for i in range(len(lb1s)):
        pos1 = hmdp.ss.sub2Positions((lb1s[i], lb2s[i]))
        pos2 = hmdp.ss.sub2Positions((ub1s[i], ub2s[i]))
        print("- lbs: [{}, {}]\n  ubs: [{}, {}]"
              .format(pos1[0], pos1[1], pos2[0], pos2[1]))


if __name__ == "__main__":
    np.random.seed(0)
    # https://stackoverflow.com/a/2891805
    np.set_printoptions(precision=3, suppress=True)
    main()
