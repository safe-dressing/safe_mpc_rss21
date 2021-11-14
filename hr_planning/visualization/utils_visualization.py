# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.

https://github.com/befelix/safe-exploration/blob/master/safe_exploration/visualization/utils_visualization.py
'''

import numpy as np
import numpy.linalg as nLa
from IPython import embed
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_FAIL(m):
    print(bcolors.FAIL + m + bcolors.ENDC)


def print_OK(m):
    print(bcolors.WARNING + m + bcolors.ENDC)


def plot_ellipsoid_2D(centroid, Q, ax, color="r", alpha=1.0):
    A = nLa.inv(Q)

    # -------------
    # https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953
    # A : (d x d) matrix of the ellipse equation in the 'center form':
    # (x-c)' * A * (x-c) = 1
    # 'centroid' is the center coordinates of the ellipse.

    # V is the rotation matrix that gives the orientation of the ellipsoid.
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # http://mathworld.wolfram.com/RotationMatrix.html
    U, D, V = nLa.svd(A)

    # x, y radii.
    rx, ry = 1./np.sqrt(D)
    # Major and minor semi-axis of the ellipse.
    dx, dy = 2 * rx, 2 * ry
    a, b = max(dx, dy), min(dx, dy)
    # Eccentricity
    # e = np.sqrt(a ** 2 - b ** 2) / a

    arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))
    if np.isnan(arccos):
        print_FAIL("arccos={}".format(arccos))
        return None, None

    # Orientation angle (with respect to the x axis counterclockwise).
    alpha_angle = arccos if arcsin > 0. else -1. * arccos
    # print -1*np.rad2deg(np.arcsin(V[0][0])), np.rad2deg(np.arccos(V[0][1]))
    # print np.rad2deg(np.arccos(V[1][0])), np.rad2deg(np.arcsin(V[1][1]))

    # Plot ellipsoid.
    ellipse2 = Ellipse(xy=centroid, width=a, height=b, edgecolor=color,
                       angle=alpha_angle, fc='None', lw=2, alpha=alpha)
    handle = ax.add_patch(ellipse2)
    return ax, handle
