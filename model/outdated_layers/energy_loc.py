import numpy as np
import tensorflow as tf
from ..utils.create_orbital_values import create_orbital_values
from datetime import datetime as dt
from .tensor_product_expansion import TensorProductExpansionLayer

from tensorflow.keras import layers
from ..layers.pairmix_layer import PairmixLayer

class EnergyLoc(layers.Layer):
    def __init__(self, name='energy_loc', **kwargs):
        super().__init__(name=name, **kwargs)


    def build(self, shape):
        #self.reduce_feature_vec_matrix = self.add_weight(name="reduce_feature_vec_matrix", shape=(14, self.F,), dtype=tf.float32, trainable=True)
        pass

    def call(self, inputs):
        # h1: shape (None,)
        h1, h2, V_n, x_i, N_mo_orb_cumsum = inputs
        N_molec = len(N_mo_orb_cumsum)

        denom = tf.zeros_like(N_mo_orb_cumsum)
        res = tf.zeros_like(denom)
        for (v_n, ampl) in V_n:
            e_1 = tf.einsum("na,nab,nb->n", v_n, h1, x_i)
            e_2 = tf.einsum("na,nb,nabcd,nc,nd->n", v_n, v_n, h2, x_i, x_i)
            molec_index = tf.repeat(tf.range(N_molec), N_mo_orb_cumsum)
            res += ampl * (tf.math.unsorted_segment_sum(e_1 + e_2), molec_index, N_molec)
            denom += ampl
        return res / denom
