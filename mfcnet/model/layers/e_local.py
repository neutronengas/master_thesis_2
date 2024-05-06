import numpy as np
import tensorflow as tf
from ..utils.create_orbital_values import create_orbital_values
from ..utils.e_local_helper import create_neighbour_states
from ..utils.calc_en import calc_en_batchwise as calc_en_V_n
from datetime import datetime as dt
from ..outdated_layers.tensor_product_expansion import TensorProductExpansionLayer

from tensorflow.keras import layers
from .pairmix_layer import PairmixLayer

class ELocalLayer(layers.Layer):
    def __init__(self, n_features, N_u, name='e_local_retry', **kwargs):
        self.N_u = N_u
        self.n_features = n_features
        super().__init__(name=name, **kwargs)


    def call(self, inputs):
        # mo_features: (batch, mo, features, angular momentum), V_n: (batch, self.N_u, 20)
        # h1: (batch, n_h1), h1_idx: (batch, n_h1, 2), h2: (batch, n_h2), h2_idx: (batch, n_h2, 4)
        mo_features, V_n, V_n_idx, h1, h1_idx, h2, h2_idx, gram = inputs
        V_n_seg = tf.gather(V_n, V_n_idx)
        loc_energy, denom, top_N_u_states = calc_en_V_n(mo_features, V_n_seg, h1, h1_idx, h2, h2_idx, self.n_features, self.N_u, gram)
       
        # TODO: validate this
        p = denom ** 2 / tf.reduce_sum(denom ** 2, axis=1)[:, None]
        energy = tf.einsum("ab,ab->a", loc_energy, p)

        V_n_out = tf.tensor_scatter_nd_update(V_n, V_n_idx, top_N_u_states)

        return energy, V_n_out