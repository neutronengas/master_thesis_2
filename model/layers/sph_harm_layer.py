import tensorflow as tf
import numpy as np
import sympy as sym
from scipy.special import binom
from tensorflow.keras import layers
from .basis_utils import spherical_harmonics

class SphHarmLayer(layers.Layer):
    def __init__(self, name='sphharm', **kwargs):
        super().__init__(name=name, **kwargs)
        
        # retrieve formulas

        x = sym.symbols("x")
        y = sym.symbols("y")
        z = sym.symbols("z")
        self.sph_harms = spherical_harmonics()

        self.sph_harm_funcs = []
        for i in range(len(self.sph_harms)):
            sph_harm_func = sym.lambdify([x, y, z], self.sph_harms[i], "tensorflow")
            self.sph_harm_funcs.append(sph_harm_func)


    def call(self, inputs):
        x = inputs[:, :, :, 0]
        y = inputs[:, :, :, 1]
        z = inputs[:, :, :, 2]
        res = tf.stack([sph_harm_func(x, y, z) for sph_harm_func in self.sph_harm_funcs], axis=-1)
        return res