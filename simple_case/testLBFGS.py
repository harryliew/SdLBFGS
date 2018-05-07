#coding:UTF-8
'''
@author: Yingkai Li, Huidong Liu
'''

from numpy import *
from lbfgs import *
import numpy as np

import matplotlib.pyplot as plt

iteration = 10000
x0 = mat([[-1.2], [1]])
result = np.log(sdlbfgs(fun, gfun, x0, iteration))
#print(*result)

n = len(result)
ax = plt.figure().add_subplot(111)
x = np.log(range(1, n+1, 1))
y = result
ax.plot(x,y, label='SdLBFGS')

x0 = mat([[-1.2], [1]])
result = np.log(sdlbfgs_old(fun, gfun, x0, iteration))
#print(*result)
z = result
ax.plot(x,z, label='SdLBFGS_old')
plt.legend()
plt.xlabel('iteration times (log)')
plt.ylabel('loss (log)')

plt.show()
