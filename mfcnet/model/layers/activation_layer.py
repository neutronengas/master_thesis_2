from tensorflow.keras import layers
import tensorflow as tf

class ActivationLayer(layers.Layer):
    def __init__(self, name='activation', **kwargs):
        super().__init__(name=name, **kwargs)

        self.alpha = self.add_weight(name="alpha", shape=(1,), dtype=tf.float32, initializer="glorot_uniform", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(1,), dtype=tf.float32, initializer="glorot_uniform", trainable=True)
        
    def call(self, inputs):
        [leq0_embeddings, leq1_embeddings, leq2_embeddings, leq3_embeddings, leq4_embeddings, leq5_embeddings] = inputs
        leq0_embeddings = self.alpha * leq0_embeddings / (1 + tf.exp(-self.beta * leq0_embeddings))
        return [leq0_embeddings, leq1_embeddings, leq2_embeddings, leq3_embeddings, leq4_embeddings, leq5_embeddings]