import tensorflow as tf
import numpy as np
import math
from scipy.special import binom
from tensorflow.keras import layers


class SphHarmLayerml(layers.Layer):
    def __init__(self, m, l, name='sphharmml', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.m = m
        self.l = l

    def call(self, inputs):
        # r: (None, 3)
        r = inputs
        r_norm = tf.norm(r, axis=1)
        l = tf.reshape(tf.cast(self.l, dtype=tf.float32), (1,))
        m = tf.reshape(tf.cast(self.m, dtype=tf.float32), (1,))
        
        prefactor = tf.reshape(tf.sqrt((self.l + 0.5) / math.pi), (1,))

        def factorial(x):
            x = tf.cast(x, dtype=tf.float32)
            return tf.exp(tf.math.lgamma(x + 1))
        
        def log_factorial(x):
            x = tf.cast(x, dtype=tf.float32)
            return tf.math.lgamma(x + 1)

        PI_ml = prefactor
        PI_ml *= tf.sqrt(factorial(self.l - self.m) / factorial(self.l + self.m))
        k = tf.range(int(tf.math.floor((self.l - self.m) / 2)) + 1)[None, :]
        k = tf.cast(k, dtype=tf.float32)
        
        # work in logarithmic scale
        fac = tf.cast(tf.math.floormod(k, 2), dtype=tf.float32)
        fac -= l * tf.math.log(2.)
        fac += log_factorial(l) - log_factorial(l - k) - log_factorial(k)
        fac += log_factorial(2 * l - 2 * k) - log_factorial(l - 2 * k) - log_factorial(l)
        fac += log_factorial(l - 2 * k) - (l - 2 * k - m)
        fac += (2 * k - 1) * (tf.math.log(r[:, 2]) - tf.math.log(r_norm))[:, None]
        fac -= m * tf.math.log(r[:, 2, None])
        fac = tf.math.exp(fac)
        PI_ml *= fac
        PI_ml = tf.reduce_sum(PI_ml, axis=1)

        if self.m == 0:
            res = PI_ml / tf.sqrt(2.)
        else:
            trigon = tf.sin if self.m < 0 else tf.cos
            m = tf.math.abs(m)
            m = tf.cast(m, dtype=tf.float32)[None, :]
            p = tf.range(tf.math.abs(self.m) + 1, dtype=tf.float32)[None, :]

            # work in logarithmic scale
            fac = log_factorial(m) - log_factorial(p) - log_factorial(m - p)
            fac += p * tf.math.log(r[:, 0, None]) 
            fac += (m - p) * tf.math.log(r[:, 1, None])
            fac += tf.math.log(trigon(m - p) * math.pi / 2)
            fac = tf.math.exp(fac)
            fac = tf.reduce_sum(fac, axis=1)
            res = PI_ml * fac
        return res