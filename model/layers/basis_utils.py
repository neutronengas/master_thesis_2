import numpy as np
import sympy as sym
from functools import partial
import math
import tensorflow as tf

from scipy.special import binom

def b(v, n, x):
    res = binom(n, v) * x ** v * (1 - x) ** (n - v)
    return res

def fcut(r, rcut):
    if rcut >= r:
        return 0
    return np.exp(-r ** 2 / (rcut ** 2 - r ** 2))

def bernstein_polys(K, gamma):
    res = [lambda r: binom(k, K - 1, np.exp(-gamma * r) * fcut(r)) for k in range(K)]
    res = None
    return res

def odd_hermite_functions(n_features):
    x = sym.symbols("x")
    
    functions = []
    func = sym.exp(-x ** 2.)
    for i in range(n_features - 1):
        func = sym.diff(func, x)
        n = 2. * i + 1
        appendix = -1 * 1 / (sym.sqrt(2 ** n * sym.factorial(n) * sym.sqrt(sym.pi))) * func * sym.exp(x ** 2 / 2)
        functions.append(sym.simplify(appendix))
        func = sym.diff(func, x)
    functions = [x] + functions
    return functions

def pi_m_l(m, l, z, r):
    fac = sym.sqrt(sym.factorial(l-m)/sym.factorial(l+m) * 1.)
    sum = 0.
    for k in range(math.floor((l - m) // 2) + 1):
        summand = (-1)**k * 2**(-l) * sym.binomial(l, k) * sym.binomial(2*l - 2*k, l)
        summand *= sym.factorial(l - 2*k) / sym.factorial(l - 2*k - m)
        summand *= z**(l - 2*k - m) / r**(2*k - l)
        #print(summand, l, m, (l - m) // 2)
        sum += summand
    return fac * sum

def spherical_harmonics():
    lm_pairs = [
        [0, 0],
        [1, -1],
        [1, 0],
        [1, 1],
        [2, -2],
        [2, -1],
        [2, 0],
        [2, 1],
        [2, 2]
    ]
    #print(x)

    x = sym.Symbol("x")
    y = sym.Symbol("y")
    z = sym.Symbol("z")

    r = sym.sqrt(y**2. + x**2. + z**2.)

    sph_harms = []
    for l, m in lm_pairs:
        abs_m = m if m > 0 else -m
        fac = sym.sqrt((2*l + 1) / (2 * math.pi)) * pi_m_l(abs_m, l, z, r)
        sum = 0
        if m == 0:
            sum += 1/math.sqrt(2)
        elif m > 0:
            for p in range(m + 1):
                summand = sym.binomial(m, p) * x**p * y**(m-p) * sym.cos((m-p) * math.pi/2)
                sum += summand
        else:
            for p in range(-m + 1):
                summand = sym.binomial(-m, p) * x**p * y**(-m-p) * sym.sin((-m-p) * math.pi/2)
                sum += summand
        sph_harms.append(fac * sum * 1. ** r)
    return sph_harms