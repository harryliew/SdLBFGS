# SdLBFGS
This project implements Stochastic damped LBFGS (SdLBFGS) [1] in PyTorch. 

## Usage:
1. Put sdlbfgs.py and sdlbfgs0.py in YOUR_PYTHON_PATH/site-packages/torch/optim. <br>
2. Open YOUR_PYTHON_PATH/site-packages/torch/optim/\_\_init__.py add the following code:
```
  from .sdlbfgs import SdLBFGS
  from .sdlbfgs0 import SdLBFGS0
  del sdlbfgs
  del sdlbfgs0
```
3. Save \_\_init__.py and restart your python. <br>
4. Just use SdLBFGS as a normal optimizer in PyTorch. <br>
Some results are presented in [2]. For any problem, please contact Huidong Liu at h.d.liew@gmail.com <br>

## References:
[1] Xiao Wang, Shiqian Ma, Donald Goldfarb, Wei Liu. "Stochastic quasi-Newton methods for nonconvex stochastic optimization." SIAM Journal on Optimization 27.2 (2017): 927-956. <br>
[2] Yingkai Li and Huidong Liu. "Implementation of Stochastic Quasi-Newton's Method in PyTorch." arXiv preprint arXiv:1805.02338, 2018.
