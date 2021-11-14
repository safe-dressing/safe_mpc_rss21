# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''
# This code is adapted from https://github.com/sawcordwell/pymdptoolbox/blob/master/src/examples/firemdp.py

from mdptoolbox import mdp
import os
import itertools
import hr_planning
import numpy as np
from hr_planning.utils_cache import read_from_memmap, save_to_memmap
from hr_planning.env_gridworld_human.discretized_state_space import DiscretizedStateSpace


class HumanMdp(object):
    def __init__(self, config_path, cache_dir, obstacles_positions=None):
        """
        Class for an MDP modeling the human behavior.
        Parameters
        ----------
        config_path: path for a config file for the DiscretizedStateSpace.
        cache_dir: directory to save the computed transition, reward, and policy.
        obstacles_positions: positions of the obstacles.
                             If None, then will use the obstacle positions from config_path.
                             If not None, then will use obstacles_positions.
        """

        self.cache_dir = cache_dir

        self.ss = DiscretizedStateSpace(config_path=config_path)
        self.config = self.ss.config

        self.pos_goal = np.array(self.config["goal_positions"])
        assert self.pos_goal.shape[0] == self.ss.n_dofs
        self.pos_goal = np.reshape(self.pos_goal, (self.ss.n_dofs, 1))
        self.sub_goal = self.ss.positions2Sub(
                tuple(np.squeeze(self.pos_goal)))
        self.ind_goal = self.ss.sub2ind(self.sub_goal)

        self.goal_reward = self.config["rewards"]["goal"]
        self.obstacle_reward = self.config["rewards"]["obstacle"]
        self.step_reward = self.config["rewards"]["step"]

        # Args override the yaml
        if obstacles_positions is None:
            obstacles_positions = self.config["obstacles_positions"]
        # Inclusive
        self.subs_obstacle = []
        self.inds_obstacle = []
        for obs in self.config["obstacles_positions"]:
            # Rectangle obstacle
            if "lbs" in obs and "ubs" in obs:
                lbs_pos = tuple(obs["lbs"])
                assert len(lbs_pos) == self.ss.n_dofs
                ubs_pos = tuple(obs["ubs"])
                assert len(ubs_pos) == self.ss.n_dofs
                lbs_sub = self.ss.positions2Sub(lbs_pos)
                ubs_sub = self.ss.positions2Sub(ubs_pos)
                ranges = []
                for i in range(self.ss.n_dofs):
                    # Inclusive
                    tmp = list(range(lbs_sub[i], ubs_sub[i] + 1))
                    ranges.append(tmp)
                self.subs_obstacle.extend(list(itertools.product(*ranges)))
            # Circle obstacle
            elif "center" in obs and "radii" in obs:
                center = np.array(tuple(obs["center"]))
                radii = obs["radii"]
                for ind in range(self.ss.n_states):
                    pos = np.array(self.ss.ind2Positions(ind))
                    assert pos.shape == center.shape
                    dist = np.linalg.norm(pos - center, ord=2)
                    if dist <= radii:
                        subs = self.ss.ind2sub(ind)
                        self.subs_obstacle.append(subs)
            else:
                raise NotImplementedError()

        for sub in self.subs_obstacle:
            ind = self.ss.sub2ind(sub)
            self.inds_obstacle.append(ind)
        assert (self.sub_goal not in self.subs_obstacle)
        assert (self.ind_goal not in self.inds_obstacle)

    def computeTransitionAndRewardArrays(self):
        """Generate the transition and reward matrices.

        Let ``S`` = number of states, and ``A`` = number of actions.

        Output (saved to cache_dir as files)
        ----------
        P = AxSxS numpy array = transition matrix.
        R = SxA numpy array = transition matrix.
        """

        P = np.zeros((self.ss.n_mps, self.ss.n_states, self.ss.n_states),
                     dtype=np.float32)
        R = np.zeros((self.ss.n_states, self.ss.n_mps), dtype=np.float32)

        for ind in range(self.ss.n_states):
            for mpi in range(self.ss.n_mps):
                R[ind, mpi] = self.step_reward
                if ind == self.ind_goal:
                    R[ind, mpi] += self.goal_reward
                if ind in self.inds_obstacle:
                    R[ind, mpi] += self.obstacle_reward

                P[mpi][ind] = np.zeros((self.ss.n_states,), dtype=np.float32)
                _, ind_prime = self.ss.transitInd(ind, mpi)
                P[mpi][ind][ind_prime] = 1.

        self.np_a_s_sPrime_2_p = P
        self.np_s_a_2_r = R

        print("Caching transition and reward matrices ...")
        path = os.path.join(self.cache_dir, "np_a_s_sPrime_2_p.dat")
        save_to_memmap(P, path)
        path = os.path.join(self.cache_dir, "np_s_a_2_r.dat")
        save_to_memmap(R, path)

    def value_iteration(self, discount=1., epsilon=0.01, max_iter=1000):
        """
        Run value iteration to compute the optimal policy.
        Parameters
        ----------
        discount: float \in (0,1], discount factor of the MDP.
        epsilon: float, tolerance of error when the algorithm converges.
        max_iter: int, maximum number of iterations to run.

        Output (saved to cache_dir as files)
        ----------
        np_s_2_a_VI = S, numpy vector = the optimal policy.
        np_s_2_v_VI = S, numpy vector = the optimal value.
        """

        print("ValueIteration:")
        ttt = mdp.ValueIteration(self.np_a_s_sPrime_2_p,
                                 self.np_s_a_2_r,
                                 discount, epsilon, max_iter)
        ttt.setVerbose()
        print("Run value iteration ...")
        ttt.run()

        print("Caching the policy ...")
        # policy = a numpy array of length |S|
        self.np_s_2_a_VI = np.array(ttt.policy, dtype=np.int32)
        path = os.path.join(self.cache_dir, "np_s_2_a_VI.dat")
        save_to_memmap(self.np_s_2_a_VI, path)
        self.np_s_2_v_VI = np.array(ttt.V, dtype=np.float32)
        path = os.path.join(self.cache_dir, "np_s_2_v_VI.dat")
        save_to_memmap(self.np_s_2_v_VI, path)

    def loadTransitionAndRewardArrays(self):
        print("loadTransitionAndRewardArray ...")
        self.np_a_s_sPrime_2_p = np.zeros((self.ss.n_mps,
                                           self.ss.n_states,
                                           self.ss.n_states),
                                          dtype=np.float32)
        path = os.path.join(self.cache_dir, "np_a_s_sPrime_2_p.dat")
        self.np_a_s_sPrime_2_p = read_from_memmap(self.np_a_s_sPrime_2_p, path)

        self.np_s_a_2_r = np.zeros((self.ss.n_states, self.ss.n_mps),
                                   dtype=np.float32)
        path = os.path.join(self.cache_dir, "np_s_a_2_r.dat")
        self.np_s_a_2_r = read_from_memmap(self.np_s_a_2_r, path)

    def loadPolicyVI(self):
        print("loadPolicyVI ...")
        self.np_s_2_a_VI = np.zeros((self.ss.n_states,), dtype=np.int32)
        path = os.path.join(self.cache_dir, "np_s_2_a_VI.dat")
        self.np_s_2_a_VI = read_from_memmap(self.np_s_2_a_VI, path)
        self.np_s_2_v_VI = np.zeros((self.ss.n_states,), dtype=np.float32)
        path = os.path.join(self.cache_dir, "np_s_2_v_VI.dat")
        self.np_s_2_v_VI = read_from_memmap(self.np_s_2_v_VI, path)

    def printPolicy(self):
        """Only works for 2D gridworld"""
        assert self.ss.n_dofs == 2
        p = np.array(self.np_s_2_a_VI).reshape(tuple(self.ss.dof_2_dims))
        print("    " + "  ".join("%2d" % y for y in range(self.ss.dof_2_dims[0])))
        print("    " + "----" * self.ss.dof_2_dims[0])
        for y in reversed(range(self.ss.dof_2_dims[1])):
            s = " %2d|," % y
            tmp = []
            for x in range(self.ss.dof_2_dims[0]):
                if (x, y) == self.sub_goal:
                    tmp.append("*".ljust(3))
                elif (x, y) in self.subs_obstacle:
                    tmp.append("o".ljust(3))
                else:
                    tmp.append(self.ss.mpi_2_str[p[x, y]].ljust(3))
            s += " ".join(tmp)
            print(s)

    def aH_2_aH_noisy(self, aH, epsilon=0.):
        """
        Add epsilon-noise to an input action.
        If sampled p < epsilon" randomly choose an action.
        If sampled p >= epsilon" choose the input action.

        Return
        ----------
        j = int, the index of the epsilon-noisy action.
        """

        # https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
        assert -1e-5 < epsilon < 1 + 1e-5
        p = np.random.random()
        j = -1
        if p < epsilon:
            j = np.random.choice(self.ss.n_mps)
        else:
            j = aH
        return j

    def rollout_ind_traj(self, ind0, min_horizon=None):
        """
        Rollout the optimal policy computed from value iteration.

        Parameters
        ----------
        ind0: int, the index of the initial state.
        min_horizon: int, the minimal number of time steps to rollout.
                     The system will only stop if
                        it has reached the goal,
                        AND the total running time >= min_horizon.
        Return 
        ----------
        ind_traj = list of int, the indices of the rollout trajectory.
        """

        s = ind0
        ind_traj = []
        ind_traj.append(s)
        while True:
            a = self.np_s_2_a_VI[s]
            s_primes = self.np_a_s_sPrime_2_p[a, s, :]
            s_prime = np.random.choice(a=len(s_primes), p=s_primes)
            ind_traj.append(s_prime)
            # Rollout till reaching the goal
            if s_prime == self.ind_goal:
                if min_horizon is None:
                    break
                else:
                    if len(ind_traj) >= min_horizon + 1:
                        break
            s = s_prime
        return ind_traj


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
    config_path = os.path.join(config_dir, "hmdp.yaml")
    cache_dir = os.path.join(module_dir, "env_gridworld_human/cache")

    hmdp = HumanMdp(config_path=config_path, cache_dir=cache_dir)
    hmdp.computeTransitionAndRewardArrays()
    hmdp.value_iteration(discount=1, epsilon=0.01, max_iter=1000)
    hmdp.printPolicy()

    hmdp2 = HumanMdp(config_path=config_path, cache_dir=cache_dir)
    hmdp2.loadTransitionAndRewardArrays()
    hmdp2.loadPolicyVI()
    assert (hmdp.np_a_s_sPrime_2_p == hmdp2.np_a_s_sPrime_2_p).all()
    assert (hmdp.np_s_a_2_r == hmdp2.np_s_a_2_r).all()
    assert (hmdp.np_s_2_v_VI == hmdp2.np_s_2_v_VI).all()
    assert (hmdp.np_s_2_a_VI == hmdp2.np_s_2_a_VI).all()
    hmdp2.printPolicy()

    aH_2_count = {}
    for i in range(1000):
        aH_noisy = hmdp.aH_2_aH_noisy(aH=0, epsilon=0.2)
        if aH_noisy not in aH_2_count:
            aH_2_count[aH_noisy] = 0
        aH_2_count[aH_noisy] += 1
    print(aH_2_count)

    t1 = hmdp.rollout_ind_traj(ind0=0, min_horizon=1)
    for i in t1:
        sub = hmdp.ss.ind2sub(i)
        print("t1,{}={}".format(i, sub))
    t2 = hmdp.rollout_ind_traj(ind0=0, min_horizon=10)
    # assert len(t2) == 11
    for i in t2:
        sub = hmdp.ss.ind2sub(i)
        print("t2,{}={}".format(i, sub))
    t3 = hmdp.rollout_ind_traj(ind0=0, min_horizon=None)
    assert t3[-1] == hmdp.ind_goal
    for i in t3:
        sub = hmdp.ss.ind2sub(i)
        print("t3,{}={}".format(i, sub))
    assert (t3 == t1)
    print("Done")
