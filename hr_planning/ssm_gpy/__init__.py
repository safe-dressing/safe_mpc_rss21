# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:51 2017

@author: tkoller
https://github.com/befelix/safe-exploration/blob/master/safe_exploration/ssm_gpy/__init__.py
"""

try:
	import GPy
except:
	raise ImportError("Subpackage ssm_gpy requires optional dependency GPy")


from .gaussian_process import *
from .gp_models_old import *
