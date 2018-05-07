# SdLBFGS
This project implements Stochastic damped LBFGS [1] in PyTorch.

Usage:

Put sdlbfgs.py and sdlbfgs0.py in YOUR_PYTHON_PATH/site-packages/torch/optim.
Open YOUR_PYTHON_PATH/site-packages/torch/optim/__init__.py add the following code:
from .sdlbfgs import SdLBFGS
from .sdlbfgs0 import SdLBFGS0
del sdlbfgs
del sdlbfgs0
Save __init__.py and restart your python.
Just use SdLBFGS as a normal optimizer in PyTorch.
For any problem, please contact Huidong Liu at h.d.liew@gmail.com

References:
[1] Wang, Xiao, et al. "Stochastic quasi-Newton methods for nonconvex stochastic optimization." SIAM Journal on Optimization 27.2 (2017): 927-956.
