import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sympy.physics.quantum.cg import CG

class CgcLayer(layers.Layer):
    def __init__(self, L, name='cgc', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.L = L

    def call(self, inputs):
        # r: (None,)
        l = {}
        for l1 in range(self.L + 1):
            for l2 in range(l1 + 1):
                l3_range = l1 + np.arange(-l2, l2 + 1)
                l3_range = [r for r in l3_range if r <= self.L]
                for l3 in l3_range:
                    m1_range = np.arange(-l1, l1 + 1)
                    m2_range = np.arange(-l2, l2 + 1)
                    M_range = np.arange(-l3, l3 + 1)
                    tens = np.fromfunction(np.vectorize(lambda m1, m2, M: (CG(l1, m1_range[m1], l2, m2_range[m2], l3, M_range[M]).doit())), 
                                       dtype=np.int32, shape=(2*l1+1, 2*l2+1, 2*l3+1))
                    l[(l1, l2, l3)] = tf.convert_to_tensor(tens, dtype=tf.float32)
                    l[(l2, l1, l3)] = tf.convert_to_tensor(tens.transpose((1, 0, 2)), dtype=tf.float32)
        return l