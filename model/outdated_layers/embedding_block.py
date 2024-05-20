import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


class EmbeddingBlock(layers.Layer):
    def __init__(self, F, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.F = F
        self.weight_init = tf.keras.initializers.GlorotUniform()

        # for now 14 different atoms are assumed
        L_init = 2
        emb_init = tf.initializers.GlorotNormal()
        
        self.leq0_embeddings = self.add_weight(name="l0_embeddings", shape=(14, self.F, 1), dtype=tf.float32, initializer=emb_init, trainable=True)
        self.leq1_embeddings = self.add_weight(name="l1_embeddings", shape=(14, self.F, 3), dtype=tf.float32, initializer=emb_init, trainable=True)
        self.leq2_embeddings = self.add_weight(name="l2_embeddings", shape=(14, self.F, 5), dtype=tf.float32, initializer=emb_init, trainable=True)
        self.leq3_embeddings = tf.zeros(shape=(14, self.F, 7))
        self.leq4_embeddings = tf.zeros(shape=(14, self.F, 9))
        self.leq5_embeddings = tf.zeros(shape=(14, self.F, 11))


        self.dense_rdm = layers.Dense(self.F, activation=activation, use_bias=True, kernel_initializer=self.weight_init)
        self.dense = layers.Dense(self.F, activation=activation, use_bias=True, kernel_initializer=self.weight_init)


    def call(self, inputs):
        Z = inputs
        # out: (None, self.no_orbitals_per_atom, self.embeddings)
        out = [tf.gather(self.leq0_embeddings, Z), 
               tf.gather(self.leq1_embeddings, Z), 
               tf.gather(self.leq2_embeddings, Z),
               tf.gather(self.leq3_embeddings, Z),
               tf.gather(self.leq4_embeddings, Z),
               tf.gather(self.leq5_embeddings, Z),
            ]
        return out