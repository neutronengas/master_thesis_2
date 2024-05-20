import tensorflow as tf
import math
import numpy as np
from scipy.special import binom
from tensorflow.keras import layers
from ..utils.cgc_multiplier import cgc_multiplier
from ..layers.sph_harm_layer import SphHarmLayer

class AtomMpnn(layers.Layer):
    def __init__(self, n_features, n_orb, activation, name='atom_mpnn', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.n_orb = n_orb # 3 in the case of STO-3g (1s, 2s, 2p)
        self.n_features = n_features
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.activation = activation
        self.cgc = cgc_multiplier(9)
        self.sph_harm_layer = SphHarmLayer()

    def build(self, shape):
        self.gamma = self.add_weight(name="gamma1", shape=(1,), dtype=tf.float32, initializer=self.initializer)
        self.w1 =[]
        for i in range(5):
            w = self.add_weight(name=f"weight_{i}", shape=(self.n_orb, self.n_orb), dtype=tf.float32, initializer=self.initializer)
            self.w1.append(w)
        #self.k = self.add_weight(name="k", shape=(self.n_features, 3, 2), dtype=tf.float32, initalizer=self.initializer)


    def call(self, inputs):
        # atom_embeddings: (None, 2, 5, features), R: (None, 10, 2, 3): C: (None, 10, 2, 5), Z, S_matrix: (None, 2, 3, 2, 3)
        ao_embeddings = inputs["ao_embeddings"]
        R = inputs["R"]
        C = inputs["C"]
        S = inputs["S"]

        # mo_features of shape (None, 10, 2, 5, features)
        mo_features = C[:, :, :, :, None] * ao_embeddings
        for w in self.w1:
            S = tf.einsum("ab,cd,kibjd->kiajc", w, w, S)
            S = self.activation(S)
        filter = tf.constant([
            [1., 1., 0.],
            [1., 1., 0.],
            [0., 0., 1.]
        ])
        S = filter[None, None, :, None, :] * S
        S = tf.repeat(S, axis=2, repeats=[1, 1, 3])
        S = tf.repeat(S, axis=4, repeats=[1, 1, 3])
        #S = tf.einsum("abcdef,c,f->abde", S, self.k, self.k)
        mo_features = tf.einsum("nmaof,naobp->nmbpf", mo_features, S)
        mo_features = tf.transpose(mo_features, perm=(3, 1, 2, 0, 4))
        mo_features = tf.math.unsorted_segment_sum(mo_features, segment_ids=[0, 0, 1, 2, 3], num_segments=9)
        mo_features = tf.transpose(mo_features, perm=(3, 1, 2, 0, 4))
        C_norm = tf.linalg.norm(C, axis=-1)[:, :, :, None, None]
        #mo_coupling = tf.concat([R[:, :, :, :, None], C_norm, tf.repeat(tf.zeros_like(C_norm), repeats=5, axis=3)], axis=-2)
        mo_coupling = self.sph_harm_layer(R)[:, :, :, :, None] * C_norm
        mo_coupling = tf.repeat(mo_coupling, repeats=self.n_features, axis=-1)
        mo_features = tf.einsum("nmapf,nmakf,kip->nmaif", mo_features, mo_coupling, self.cgc)
        mo_features = tf.reduce_sum(mo_features, axis=2)
        # combining the two s-orbitals, padding the 5 components for l=2 and repeating the l=1-component 3 times
        # return shape: (batch, mo, features, ang mom)
        mo_features = tf.transpose(mo_features, perm=(0, 1, 3, 2))
        return mo_features