import numpy as np
import tensorflow as tf
from ..utils.create_orbital_values import create_orbital_values
from datetime import datetime as dt
from .tensor_product_expansion import TensorProductExpansionLayer

from tensorflow.keras import layers
from ..layers.pairmix_layer import PairmixLayer

class ELocalLayer(layers.Layer):
    def __init__(self, N_u, h1_threshold, h2_threshold, name='e_local', **kwargs):
        self.h1_threshold = h1_threshold
        self.h2_threshold = h2_threshold
        self.N_u = N_u
        self.V_n = None
        super().__init__(name=name, **kwargs)

    def build(self, shape):
        self.w = self.add_weight(name="w", shape=(14, 14,), dtype=tf.float32, trainable=True)
        pass


    def call(self, inputs):
        # V_n_hf: (None, self.N_u, N_orb)
        molec_feature_vectures, h1, h2, V_n_hf = inputs
        batch_size = len(molec_feature_vectures)
        N_orb = len(h1[0])

        if self.V_n is None:
            self.V_n = V_n_hf

        def calc_ampl(molec_feature_vectors):
            # molec_feature_vectors: (N, n_equiv, n_channels):
            mtcs = tf.einsum("abc,bj,ijc->aic", molec_feature_vectors, self.w + tf.transpose(self.w), molec_feature_vectors)
            det = tf.linalg.det(tf.reduce_sum(mtcs, axis=-1))
            return det

        def e_local_vectorwise(x_i_vec, h1, h2, molec_feature_vectors, n_orb):

            h1 = tf.reshape(h1, n_orb ** 2)
            h2 = tf.reshape(h2, n_orb ** 4)

            x_i_vec = x_i_vec[:, None]

            c_dag_c = tf.eye(n_orb, dtype=tf.int32)
            c_dag_c = tf.reshape(c_dag_c[:, None, :] - c_dag_c[:, :, None], shape=(n_orb, n_orb ** 2))
            states_1 = c_dag_c + x_i_vec
            is_valid_state_1_mask = tf.math.reduce_max(states_1, axis=0) - tf.math.reduce_min(states_1, axis=0) == 1
            above_threshold_1_mask = h1 > self.h1_threshold
            states_1 = tf.boolean_mask(states_1, tf.logical_and(is_valid_state_1_mask, above_threshold_1_mask), axis=1)  
            ampls_1 = molec_feature_vectors[:, :, None] * states_1[:, None, :]
            ampls_1 = calc_ampl(ampls_1)
            e_loc_1 = tf.reduce_sum((tf.boolean_mask(h1, tf.logical_and(above_threshold_1_mask, is_valid_state_1_mask)) * ampls_1) ** 2)

            c_dag_c_dag_c_c = tf.reshape(c_dag_c[:, None, :] - c_dag_c[:, :, None], (n_orb, n_orb ** 4))       
            states_2 = c_dag_c_dag_c_c + x_i_vec
            is_valid_state_2_mask = tf.math.reduce_max(states_2, axis=0) - tf.math.reduce_min(states_2, axis=0) == 1
            above_threshold_2_mask = h2 > self.h2_threshold
            states_2 = tf.boolean_mask(states_2, tf.logical_and(is_valid_state_2_mask, above_threshold_2_mask), axis=1)
            ampls_2 = molec_feature_vectors[:, :, None] * states_2[:, None, :]
            ampls_2 = tf.reduce_sum(ampls_2, axis=(0, 1))
            e_loc_2 = tf.reduce_sum((tf.boolean_mask(h2, tf.logical_and(above_threshold_2_mask, is_valid_state_2_mask)) * ampls_2) ** 2)

            e_loc = e_loc_1 + e_loc_2
            
            p_x_i = tf.gather(molec_feature_vectors, x_i_vec)
            p_x_i = calc_ampl(p_x_i) ** 2 

            new_V_n = tf.gather(states_2, tf.math.top_k(ampls_2, self.N_u).indices)
            new_V_n_ampls = tf.gather(ampls_2, tf.math.top_k(ampls_2, self.N_u).indices)

            return e_loc, p_x_i, new_V_n, new_V_n_ampls

        self.V_n = tf.reshape(self.V_n, (len(self.V_n) * self.N_u, len(self.V_n[0, 0])))

        e_loc, p_x_i, new_V_n, new_V_n_ampls = tf.map_fn(lambda x: e_local_vectorwise(x[0], x[1], x[2], x[3], len(self.V_n[0])), (self.V_n, h1, h2, molec_feature_vectures), dtype=(tf.int32, tf.int32))

        e_loc = tf.reshape(e_loc, (len(self.V_n) // self.N_u, self.N_u))
        p_x_i = tf.reshape(p_x_i, (len(self.V_n) // self.N_u, self.N_u))
        e_loc = tf.math.reduce_sum(e_loc * p_x_i, axis=1) / tf.math.reduce_sum(p_x_i, axis=1)

        new_V_n_ampls = tf.reshape(new_V_n_ampls, (batch_size, self.N_u ** 2))
        new_V_n = tf.reshape(new_V_n, (batch_size, self.N_u ** 2, N_orb))

        def shape_V_n_new(ampls_2, V_n_new, N_u):
            # ampls of shape (N ** 2), V_n_new of shape (N ** 2, N_orb)
            _, top_k_indices = tf.math.top_k(ampls_2, k=N_u)
            V_n_filtered = tf.gather(V_n_new, top_k_indices)
            return V_n_filtered
        
        new_V_n = tf.map_fn(lambda x: shape_V_n_new(x[0], x[1], self.N_u), (new_V_n_ampls, new_V_n))
        self.V_n = new_V_n

        return e_loc