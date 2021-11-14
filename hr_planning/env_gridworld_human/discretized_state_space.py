# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import hr_planning
import numpy as np
import yaml
import os
import itertools


class DiscretizedStateSpace(object):
    def __init__(self, config_path):
        """
        Class for a discretized state space.
        Parameters
        ----------
        config_path: path for a config file.
                     This config file specifies:
                     1. The continuous state space with discretization information.
                     2. The goal position that the human is trying to reach.
                     3. Obstacle occupied positions.
                     4. MDP reward.
        """

        # The ndarray data format style:
        # Note that numpy default is C style
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel_multi_index.html
        # But for armadillo in C++, the default is F.
        # http://arma.sourceforge.net/docs.html#sub2ind
        # Here we choose to use C
        self.ndarray_style = 'C'

        self.config = None
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # 1. Human state space
        # ind (index) = 1D integer,
        # e.g., 1 means the 1st grid in the 2D map.
        # sub (subscript) = list of coordinates,
        # e.g., (1,1) means the grid coordinate (1,1) in the 2D map.
        # positions = list,
        # e.g., (0.5,0.2) means the exact position in the continuous state space.

        # n_dofsx0
        self.pos_lbs = np.array(
            self.config["state_space"]["position"]["lower_limits"],
            dtype=np.float32)
        # n_dofsx0
        self.pos_ubs = np.array(
            self.config["state_space"]["position"]["upper_limits"],
            dtype=np.float32)
        # n_dofsx0
        self.dof_2_dims = np.array(
            self.config["state_space"]["position"]["num_discretizations"],
            dtype=np.int32)
        self.n_dofs = self.pos_lbs.shape[0]
        assert self.n_dofs == self.pos_ubs.shape[0]
        assert self.n_dofs == self.dof_2_dims.shape[0]
        for i in range(self.n_dofs):
            assert self.pos_lbs[i] <= self.pos_ubs[i] - 1e-5
        self.ind_2_center_by_dof = []
        self.ind_2_radii_by_dof = []
        for i in range(self.n_dofs):
            samples, step = np.linspace(
                    start=self.pos_lbs[i], stop=self.pos_ubs[i],
                    num=self.dof_2_dims[i], endpoint=True, retstep=True,
                    dtype=np.float32)
            self.ind_2_center_by_dof.append(samples)
            self.ind_2_radii_by_dof.append(step / 2.)

        self.ind_2_bin_by_dof = []
        for i in range(self.n_dofs):
            tmp = np.zeros((self.dof_2_dims[i] - 1))
            for j in range(1, self.dof_2_dims[i]):
                b = (self.ind_2_center_by_dof[i][j]
                     + self.ind_2_center_by_dof[i][j-1]) / 2.
                tmp[j - 1] = b
            self.ind_2_bin_by_dof.append(tmp)

        self.n_states = np.prod(self.dof_2_dims)
        assert (self.n_states < np.iinfo(np.int32).max)
        self.n_states = int(self.n_states.astype(np.int32))

        # 2. Human action space
        # mpi (motion primitive index) = integer.
        # sub (motion primitive) = list of coordinates.
        self.mpi_2_sub = []
        self.mpi_2_str = []
        for i in self.config["action_space"]["motion_primitives"]:
            self.mpi_2_sub.append(i["sub"])
            self.mpi_2_str.append(i["str"])
        self.mpi_2_sub = np.array(self.mpi_2_sub)
        assert self.mpi_2_sub.shape[1] == self.n_dofs
        self.n_mps = self.mpi_2_sub.shape[0]

    def ind2Positions(self, ind):
        """int -> tuple"""
        sub = self.ind2sub(ind)
        return self.sub2Positions(sub)

    def positions2Ind(self, positions):
        """tuple -> int"""
        subscripts = self.positions2Sub(positions)
        return self.sub2ind(subscripts)

    def sub2Positions(self, sub):
        """tuple -> tuple"""
        pos = [self.ind_2_center_by_dof[jt][i] for jt, i in enumerate(sub)]
        return tuple(pos)

    def positions2Sub(self, positions):
        """tuple -> tuple"""
        subscripts = []
        for i in range(self.n_dofs):
            digit = np.digitize(x=positions[i],
                                bins=self.ind_2_bin_by_dof[i])
            subscripts.append(digit)
        return tuple(subscripts)

    def ind2sub(self, ind):
        """int -> tuple"""
        assert (ind < self.n_states and ind >= 0)
        return np.unravel_index(
            indices=ind, shape=tuple(self.dof_2_dims),
            order=self.ndarray_style)

    def sub2ind(self, subscripts):
        """tuple -> int"""
        assert len(subscripts) == self.n_dofs
        for i in range(len(subscripts)):
            assert 0 <= subscripts[i] < self.dof_2_dims[i]
        return np.ravel_multi_index(
            multi_index=subscripts, dims=tuple(self.dof_2_dims),
            order=self.ndarray_style)

    def clampSubscripts(self, sub):
        """tuple -> (out_of_limits - bool, sub_new - tuple)"""
        uppers = [self.dof_2_dims[i]-1 for i in range(self.n_dofs)]
        lowers = [0 for i in range(self.n_dofs)]
        sub_new = np.clip(sub, lowers, uppers)
        out_of_limits = True
        if (sub == sub_new).all():
            out_of_limits = False
        return out_of_limits, tuple(sub_new)

    def transitInd(self, ind, mpi):
        """int, int -> out_of_limits - bool, int"""
        subscripts = np.array(self.ind2sub(ind))
        mp = np.array(self.mpi_2_sub[mpi])
        subscripts_prime = subscripts + mp
        out_of_limits, sub_new = self.clampSubscripts(subscripts_prime)
        return out_of_limits, self.sub2ind(sub_new)


if __name__ == '__main__':
    config_dir_name = "config_2d_simple"

    path = os.path.abspath(hr_planning.__file__)
    module_dir = os.path.split(path)[0]
    config_dir = os.path.join(
            module_dir,
            "env_gridworld_human/" + config_dir_name)
    config_path = os.path.join(config_dir, "hmdp.yaml")
    ss = DiscretizedStateSpace(config_path=config_path)

    position = (0., 0.25)
    print(ss.positions2Sub(position))

    # Test the functions
    for ind in range(ss.n_states):
        sub = ss.ind2sub(ind)
        ind2 = ss.sub2ind(sub)
        assert ind == ind2

        positions = ss.ind2Positions(ind)
        positions2 = ss.sub2Positions(sub)
        print("{}={}={}".format(ind, sub, positions))
        assert ((np.array(positions) - np.array(positions2) < 1e-10).all())

        ind3 = ss.positions2Ind(positions)
        assert (ind == ind3)

        sub2 = ss.positions2Sub(positions2)
        assert sub == sub2

    coord_to_test = []
    for i in range(ss.n_dofs):
        tmp = np.linspace(ss.pos_lbs[i]-1, ss.pos_ubs[i]+1,
                          num=50, endpoint=True)
        coord_to_test.append(list(tmp))
    for positions in itertools.product(*coord_to_test):
        ind = ss.positions2Ind(positions)
        sub = ss.positions2Sub(positions)
        # print("{}={}={}".format(ind, sub, positions))

        sub2 = ss.ind2sub(ind)
        ind2 = ss.sub2ind(sub)
        assert ind2 == ind
        assert sub == sub2

        positions = ss.ind2Positions(ind)
        positions2 = ss.sub2Positions(sub)
        assert ((np.array(positions) - np.array(positions2) < 1e-10).all())

        ind3 = ss.positions2Ind(positions)
        assert (ind == ind3)

        sub3 = ss.positions2Sub(positions2)
        assert sub3 == sub2

    # Test transitions
    for ind in range(ss.n_states):
        for mpi in range(ss.n_mps):
            out_of_limits, ind_prime = ss.transitInd(ind, mpi)
            print("{}+{}={}".format(
                ss.ind2sub(ind), ss.mpi_2_sub[mpi], ss.ind2sub(ind_prime)))
