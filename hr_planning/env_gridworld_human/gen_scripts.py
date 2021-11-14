# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import os
import hr_planning
import math

starting_seed = 0
reps = 30
n_files_to_split = 2


def main():
    """
    Generate .sh files for running the benchmark
    """

    run_ca = []
    run_ca_si = []
    run_greedy = []

    for hmdp_name in ["hmdp1.yaml", "hmdp2.yaml", "hmdp3.yaml", "hmdp4.yaml", "hmdp5.yaml"]:
        for mode in ["CA", "CA_SI"]:
            # Start from the same seed for all modes
            seed = starting_seed
            s = "python3 run_mpc_iterations_experiments.py"
            if mode == "greedy":
                s = "python3 run_mpc_greedy_iterations_experiments.py"
            for task in ["coll_avoid"]:
                for pH_mode in ["pH_indep_pR", "pH_avoid_pR", "pH_move_to_pR"]:
                    for i in range(reps):
                        x = s
                        if mode != "greedy":
                            x += " --pR_mode=" + mode
                        x += " --task=" + task
                        x += " --pH_mode=" + pH_mode
                        x += " --seed=" + str(seed)
                        x += " --hmdp_name=" + hmdp_name
                        seed += 1
                        if mode == "greedy":
                            run_greedy.append(x)
                        elif mode == "CA":
                            run_ca.append(x)
                        elif mode == "CA_SI":
                            run_ca_si.append(x)
                        else:
                            raise ValueError()

    path = os.path.abspath(hr_planning.__file__)
    module_dir = os.path.split(path)[0]

    split_and_write(module_dir=module_dir,
                    file_name="run_ca",
                    cmds=run_ca)

    split_and_write(module_dir=module_dir,
                    file_name="run_ca_si",
                    cmds=run_ca_si)

    if len(run_greedy) > 0:
        split_and_write(module_dir=module_dir,
                        file_name="run_greedy",
                        cmds=run_greedy)


def split_and_write(module_dir, file_name, cmds):
    # cmds = [1, 2, 3, 4, 5]
    cmds_per_file = math.ceil(len(cmds) / n_files_to_split)
    chunks = [cmds[x:x+cmds_per_file] for x in range(
        0, len(cmds), cmds_per_file)]
    assert len(chunks) == n_files_to_split
    count = 0
    for x in chunks:
        count += len(x)
        assert len(x) <= cmds_per_file
        assert len(x) >= cmds_per_file - 1
    assert count == len(cmds)
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            tmp = set(chunks[i]).intersection(set(chunks[j]))
            assert len(tmp) == 0
    for i, chunk in enumerate(chunks):
        tmp = file_name + "_" + str(i) + ".sh"
        path = os.path.join(module_dir, tmp)
        with open(path, 'w') as the_file:
            the_file.write("#!/bin/sh\n")
            for x in chunk:
                the_file.write(x + "\n")


if __name__ == "__main__":
    main()
