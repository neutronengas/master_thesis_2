

def pi_m_l(self, m, l, r):
    r_norm = np.linalg.norm(r)
    z = r[-1]
    k_range = np.arange(int(np.floor((l - m) / 2) + 1) + 1)
    res = -1 ** k_range * binom(l, k_range) * binom(2 * l - 2 * k_range, l) 
    res *= factorial(l - 2 * k_range) / factorial(l - 2 * k_range - m) 
    res *= ((z / r_norm) ** (l - 2 * k_range)) / (z ** m)
    res = res.sum()
    res /= 2 ** l    
    res *= np.sqrt(factorial(l - m) / factorial(l + m))
    return res

def Y_m_l(self, m, l, r):
    [x, y, _] = r
    res = np.sqrt((l + 0.5) / np.pi) * self.pi_m_l(m, l, r)
    if m == 0:
        res /= np.sqrt(2)
        return res 
    trigon = np.sin if m < 0 else np.cos
    m = np.abs(m)
    p_range = np.arange(m + 1)
    fac = binom(m, p_range) * x ** p_range * y ** (m - p_range)
    fac *= trigon(np.pi / 2 * (m - p_range))
    res *= fac.sum()
    return res