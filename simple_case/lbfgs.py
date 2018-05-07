# coding:UTF-8

from numpy import *
from function import *

def sdlbfgs(fun, gfun, x0, maxk):
    result = []
    rho = 0.55
    sigma = 0.4
    delta = 1

    H0 = eye(shape(x0)[0])

    rhos = []
    s = []
    y = []
    m = 6

    k = 1
    gk = mat(gfun(x0))
    dk = -H0 * gk
    while (k < maxk):
        n = 0
        mk = 0
        gk = mat(gfun(x0))
        dk = dk / linalg.norm(dk)
        x = x0 + dk / (k ** 0.5)

        if k > m:
            s.pop(0)
            y.pop(0)

        sk = x - x0
        s.append(sk)
        yk = gfun(x) - gk
        sy = asscalar(sk.T * yk)
        if (sy == 0): 
            break
        gamma = max( asscalar(yk.T * yk) / sy, delta)
        s_H_inv_s = asscalar(gamma * sk.T * sk)
        if (sy < 0.25 * s_H_inv_s):
            theta = 0.75 * s_H_inv_s / (s_H_inv_s - sy)
        else:
            theta = 1

        yk_bar = theta * yk + (1-theta) * gamma * sk
        rho0 = 1 / (sk.T * yk_bar)
        rhos.append(rho0)
        y.append(yk_bar)

        t = len(s)
        qk = gfun(x)
        a = []
        u = gk

        for i in range(t):
            alpha = (s[t - i - 1].T * qk) / (y[t - i - 1].T * s[t - i - 1])
            qk = qk - alpha[0, 0] * y[t - i - 1]
            a.append(alpha[0, 0])
        r = H0 * qk

        for i in range(t):
            beta = (y[i].T * r) / (y[i].T * s[i])
            r = r + s[i] * (a[t - i - 1] - beta[0, 0])

        if (yk.T * sk > 0):
            dk = -r

        k = k + 1
        x0 = x
        result.append(fun(x0))

    return result

def sdlbfgs_old(fun, gfun, x0, maxk):
    result = []
    rho = 0.55
    sigma = 0.4
    delta = 1

    H0 = eye(shape(x0)[0])

    rhos = []
    s = []
    y = []
    m = 6

    k = 1
    gk = mat(gfun(x0))
    dk = -H0 * gk
    while (k < maxk):
        n = 0
        mk = 0
        gk = mat(gfun(x0))
        x = x0 + dk / (k ** 0.5)

        if k > m:
            s.pop(0)
            y.pop(0)

        sk = x - x0
        s.append(sk)
        yk = gfun(x) - gk
        sy = asscalar(sk.T * yk)
        if (sy == 0): 
            break
        gamma = max( asscalar(yk.T * yk) / sy, delta)
        H0 = 1 / gamma * H0
        s_H_inv_s = asscalar(gamma * sk.T * sk)
        if (sy < 0.25 * s_H_inv_s):
            theta = 0.75 * s_H_inv_s / (s_H_inv_s - sy)
        else:
            theta = 1

        yk_bar = theta * yk + (1-theta) * gamma * sk
        rho0 = 1 / (sk.T * yk_bar)
        rhos.append(rho0)
        y.append(yk_bar)

        t = len(s)
        qk = gfun(x)
        a = []
        u = gk

        for i in range(t):
            alpha = (s[t - i - 1].T * qk) / (y[t - i - 1].T * s[t - i - 1])
            qk = qk - alpha[0, 0] * y[t - i - 1]
            a.append(alpha[0, 0])
        r = H0 * qk

        for i in range(t):
            beta = (y[i].T * r) / (y[i].T * s[i])
            r = r + s[i] * (a[t - i - 1] - beta[0, 0])

        if (yk.T * sk > 0):
            dk = -r

        k = k + 1
        x0 = x
        result.append(fun(x0))

    return result
