import numpy as np
from sympy.physics.quantum.cg import CG
import tensorflow as tf

def cgc_multiplier(n_ang_mom):
    lm_pairs = [
        [0, 0],
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

    def cg_tensor(i, j, k):
        l1, m1 = lm_pairs[i]
        l2, m2 = lm_pairs[k]
        l3, m3 = lm_pairs[j]

        return np.asarray(CG(l1, m1, l2, m2, l3, m3).doit())
    
    tens = np.fromfunction(np.vectorize(cg_tensor), (n_ang_mom, n_ang_mom, n_ang_mom), dtype=np.int32)
    tens = tf.convert_to_tensor(tens, dtype=tf.float32)
    return tens