import numpy as np
from scipy.special import factorial, binom

def pi_m_l(r, m, l):
    z = r[:, 2]
    r_norm = np.linalg.norm(r, axis=1)
    res = np.sqrt(factorial(l - m) / factorial(l + m))
    k_range = np.arange(np.floor((l - m) / 2) + 1)
    res *= (1 - 2 * (k_range % 2)) * 2 ** (-l)
    res *= binom(l, k_range) * binom(2 * l - 2 * k_range, l)
    res *= (factorial(l - 2 * k_range) / factorial(l - 2 * k_range - m))
    res = res * r_norm[:, None] ** (2 * k_range[None, :] - l) * (z[:, None] ** (l - 2 * k_range[None, :] - m))
    res = res.sum(axis=1)
    return res


def Y_m_l(r, m, l):
    x = r[:, 0, None]
    y = r[:, 1, None]
    res = np.sqrt((2 * l + 1) / (2 * np.pi)) 
    res *= pi_m_l(r, np.abs(m), l)
    res *= 1 / np.sqrt(2)
    p_range = np.arange(np.abs(m) + 1.)
    p_sum = np.sqrt(2) * binom(np.abs(m), p_range) 
    p_sum = p_sum * x ** p_range * y ** (np.abs(m) - p_range)
    p_sum *= float(m < 0) * np.sin((np.abs(m) - p_range) * np.pi / 2) + float(m > 0) * np.cos((np.abs(m) - p_range) * np.pi / 2)
    p_sum = (p_sum.sum(axis=1)) ** float(m != 0)
    res = res * p_sum
    return res

def Y_l(r, l):
    m_range = np.arange(-l, l + 1)
    res = [Y_m_l(r, m, l) for m in m_range]
    res = np.array(res).T
    return res