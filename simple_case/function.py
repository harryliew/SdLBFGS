#coding:UTF-8
'''

@author: zhaozhiyong
'''

from numpy import *

#fun
def fun(x):
    return 100 * (x[0,0] ** 2 - x[1,0]) ** 2 + (x[0,0] - 1) ** 2

#gfun
def gfun(x):
    result = zeros((2, 1))
    result[0, 0] = 400 * x[0,0] * (x[0,0] ** 2 - x[1,0]) + 2 * (x[0,0] - 1)
    result[1, 0] = -200 * (x[0,0] ** 2 - x[1,0])
    return result
