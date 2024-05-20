import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .layers.embedding_layer import EmbeddingLayer
from .outdated_layers.cgc_layer import CgcLayer
#from .layers.interaction_block import InteractionBlock
from .layers.interaction_layer import InteractionLayer
#from .layers.output_block import OutputBlock
#from .layers.output_layer import OutputLayer
from .layers.e_local import ELocalLayer
from .activations import swish

class MFCNet(tf.keras.Model):
    def __init__(self, n_features, N_u, n_orb, k, n_mlp, l_max, n_ang_mom, num_interaction_blocks, activation=swish, output_init='zeros', name='dmnet', **kwargs):
        super().__init__(name=name, **kwargs)
        # hard-coded for cc-pvdz basis
        self.n_features = n_features
        self.N_u = N_u
        self.n_orb = n_orb
        self.k = k
        self.n_mlp = n_mlp
        self.l_max = l_max

        self.embedding_block = EmbeddingLayer(n_orb=n_orb, n_features=n_features, activation=activation)
        self.int_layers = []
        for _ in range(num_interaction_blocks):
            int_layer = InteractionLayer(n_features=n_features, n_orb=n_orb, n_ang_mom=n_ang_mom, lmax=self.l_max, n_mlp=self.n_mlp, k=self.k)
            self.int_layers.append(int_layer)
        self.output_layer = ELocalLayer(n_features=n_features, N_u=N_u)


    def call(self, inputs, V_n):
        inputs = inputs[0]
        R = inputs['R']
        C = inputs['C']
        Z = inputs['Z']
        S = inputs['S']
        h1 = inputs['h1']
        h1_idx = inputs['h1_idx']
        h2 = inputs['h2']
        h2_idx = inputs['h2_idx']
        #V_n = inputs['V_n']
        V_n_idx = inputs['idx']
        coupl = inputs['coupl']
        mo_neighbours_i = inputs['mo_neighbours_i']
        mo_neighbours_j = inputs['mo_neighbours_j']
        gram = inputs['gram']

        mo_features = self.embedding_block((R, C, Z, S))
        for layer in self.int_layers:
            mo_features = layer((mo_features, coupl, mo_neighbours_i, mo_neighbours_j))
        mo_features = tf.concat([mo_features, mo_features], axis=1)
        energy, V_n_out = self.output_layer((mo_features, V_n, V_n_idx, h1, h1_idx, h2, h2_idx, gram))
        return energy, V_n_out