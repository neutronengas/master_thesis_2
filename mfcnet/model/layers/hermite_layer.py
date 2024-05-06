import tensorflow as tf
import math
import numpy as np
from .basis_utils import odd_hermite_functions
import sympy as sym
from tensorflow.keras import layers

class HermiteLayer(layers.Layer):
    def __init__(self, n_features, name='hermite', **kwargs):
        super().__init__(name=name, **kwargs)
        
        # retrieve formulas
        self.hermite_formulas = odd_hermite_functions(n_features)

        # convert to tensorflow formulas
        x = sym.symbols("x")
        self.hermite_funcs = []
        for i in range(len(self.hermite_formulas)):
            hermite_func = sym.lambdify([x], self.hermite_formulas[i], "tensorflow")
            self.hermite_funcs.append(hermite_func)


    def call(self, inputs):
        coeffs = inputs

        res = [func(coeffs) for func in self.hermite_funcs]
        res = tf.stack(res, axis=-1)

        return res