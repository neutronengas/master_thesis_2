import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

class InteractionBlock(layers.Layer):
    def __init__(self, emb_size, num_transforms, no_orbitals_per_atom=14, activation=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.num_transforms = num_transforms
        self.activation = activation
        self.no_orbitals_per_atom = no_orbitals_per_atom
        self.dense_1 = Dense(emb_size, activation='relu')
        self.dense_2 = Dense(emb_size, activation='relu')
        self.dense_3 = Dense(emb_size, activation='relu')

    def build(self, shape):
        self.transform_mat_weights = self.add_weight(name="mat_weight", shape=(self.no_orbitals_per_atom, self.emb_size, self.emb_size), dtype=tf.float32, trainable=True)
        self.transform_biases = self.add_weight(name="bias", shape=(self.no_orbitals_per_atom, self.emb_size,), dtype=tf.float32, trainable=True)


    def call(self, inputs):
        # out: (None, self.no_orbitals_per_atom, self.emb_size); edges: (None, 3, self.emb_size), R: (None, 3); edge_id_i: (None,), edge_id_j: (None,)
        out, R, edge_id_i, edge_id_j = inputs
        msg = tf.gather(out, edge_id_i) - tf.gather(out, edge_id_j)
        K = tf.einsum("ij,kj->ki", tf.eye(3), R)
        msg = msg / tf.norm(tf.gather(R, edge_id_i) - tf.gather(R, edge_id_j), axis=1)[:, None, None]
        msg = self.dense_1(msg)
        msg = tf.einsum("ikj,nij->nik", self.transform_mat_weights, msg)
        msg = msg + self.transform_biases
        msg = self.activation(msg)
        #print(edge_id_i.shape)
        msg = tf.math.unsorted_segment_sum(msg, edge_id_j, num_segments=len(out))
        msg = self.activation(msg)
        #msg = self.dense_2(msg)
        out = out + msg
        out = self.activation(out)
        out = self.dense_3(out)
        return out