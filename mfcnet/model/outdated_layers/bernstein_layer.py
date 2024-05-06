import tensorflow as tf
import math
import numpy as np
from scipy.special import binom
from tensorflow.keras import layers

class BernsteinLayer(layers.Layer):
    def __init__(self, K, r_cut, name='bernstein', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.K = K
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.r_cut = r_cut

        binom_coeffs = binom(K - 1, np.arange(K))
        self.binom_coeffs = tf.convert_to_tensor(binom_coeffs, dtype=tf.float32)

    def build(self, shape):
        self.gamma = self.add_weight(name="gamma1", shape=(1,), dtype=tf.float32, initializer=self.initializer)


    def call(self, inputs):
        # r: (None,)
        r = inputs[:, None]
        r = tf.repeat(r, self.K, axis=1)
        inp = tf.exp(-self.gamma ** 2 * r)
        tf.debugging.check_numerics(self.gamma, "gamma is not finite")
        K_range = tf.range(self.K, dtype=tf.float32)
        b = tf.gather(self.binom_coeffs, tf.range(self.K)) * inp ** K_range * (1 - inp) ** (self.K - 1. - K_range)
        
        fcut = tf.math.cos((r / self.r_cut) ** 1.5 * math.pi ** 1.5) 
        res = b * fcut * tf.cast(tf.math.less_equal(r, self.r_cut), dtype=tf.float32)
        return res