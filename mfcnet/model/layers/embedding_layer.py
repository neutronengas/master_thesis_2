import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from ..layers.hermite_layer import HermiteLayer
from ..layers.tensor_product_contraction import TensorProductContractionLayer
from ..layers.atom_mpnn import AtomMpnn


class EmbeddingLayer(layers.Layer):
    def __init__(self, n_orb, n_features, activation, name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_features = n_features
        self.atom_mpnn = AtomMpnn(n_orb=n_orb, n_features=n_features, activation=activation)

        self.weight_init = tf.keras.initializers.GlorotUniform()
        self.atom_embeddings = self.add_weight(name="atoms_embeddings", shape=[14, 3, self.n_features], dtype=tf.float32, initializer=self.weight_init, trainable=True)
    
    def call(self, inputs):
        # keys: C: (None, 10, 2, 5), Z_ao: (None, 2), S: (None, 2, 3, 2, 3)
        R, C, Z_ao, S = inputs
        
        ao_embeddings = tf.gather(self.atom_embeddings, Z_ao)
        ao_embeddings = tf.stack([ao_embeddings for _ in range(len(C[0]))], axis=1)
        ao_embeddings = tf.repeat(ao_embeddings, [1, 1, 3], axis=-2)
        inp = {
            "ao_embeddings": ao_embeddings,
            "R": R,
            "C": C,
            "Z": Z_ao,
            "S": S
        }
        mo_features = self.atom_mpnn(inp)

        return mo_features