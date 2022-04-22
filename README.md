## Provably Safe and Efficient Motion Planning with Uncertain Human Dynamics

![Safe robot-assisted dressing](/fig.png)

* This is the code for the RSS'21 paper:
  * Provably Safe and Efficient Motion Planning with Uncertain Human Dynamics
  * Website: https://safe-dressing.github.io/
  * PDF: http://www.roboticsproceedings.org/rss17/p050.pdf
  * Robotics: Science and Systems 2021
  * Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
* This code is based on the code: [`safe-exploration`](https://github.com/befelix/safe-exploration) developed for the paper `T. Koller, F. Berkenkamp, M. Turchetta, A. Krause, Learning-based Model Predictive Control for Safe Exploration in Proc. of the Conference on Decision and Control (CDC), 2018`

## Run simulated experiment (reproducing `Table 2` and `Figure 6` in the paper)
### System
* Tested on `Ubuntu 20.04` with `Python 3.8.10`

### Installation
* `export PYTHONPATH=$PYTHONPATH:/home/.../safe_mpc_rss21/`
* `sudo apt install python3-pip`
* `sudo pip3 install numpy`
* `sudo pip3 install ipython`
* `sudo pip3 install matplotlib`
* `sudo pip3 install scipy`
* `sudo pip3 install casadi`
* `sudo pip3 install sdeint`
* `sudo pip3 install pymdptoolbox`
* `sudo pip3 install GPy`
* `sudo pip3 install sklearn`

### Run experiments and get results (`Table 2` and `Figure 6`)
* `cd /home/.../safe_mpc_rss21/hr_planning/`
* Safe MPC that only does collision avoidance
  * `./run_ca_0.sh`
  * `./run_ca_1.sh`
* Safe MPC that does collision avoidance OR safe impact
  * `./run_ca_si_0.sh`
  * `./run_ca_si_1.sh`
* Result processing
  * `python3 result_per_map.py`
  * `python3 result_all_maps.py`
    * This should print the latex code for `Table 2` in the paper.
* Safe MPC in robot-assisted dressing in 2D
  * `python3 run_mpc_iterations_experiments.py --pR_mode=CA --task=dressing_2d --pH_mode=pH_indep_pR --seed=5 --hmdp_name=hmdp.yaml`


## Contact
* shenli@mit.edu

## Citation
```
@INPROCEEDINGS{Li-RSS-21,
    AUTHOR    = {Shen Li AND Nadia Figueroa AND Ankit Shah AND Julie A. Shah},
    TITLE     = {{Provably Safe and Efficient Motion Planning with Uncertain Human Dynamics}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2021},
    ADDRESS   = {Virtual},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2021.XVII.050},
}
```