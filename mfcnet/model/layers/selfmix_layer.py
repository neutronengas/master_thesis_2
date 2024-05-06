from tensorflow.keras import layers
import tensorflow as tf

from ..layers.tensor_product_contraction import TensorProductContractionLayer



class SelfmixLayer(layers.Layer):
    def __init__(self, n_features, name='selfmix', **kwargs):
        super().__init__(name=name, **kwargs)

        self.tp_layer = TensorProductContractionLayer()
        self.n_features = n_features

    def build(self, shape):
        self.k = self.add_weight(name="k", shape=(self.n_features, 3), dtype=tf.float32, initializer="glorot_uniform", trainable=True)
        self.k = tf.repeat(self.k, repeats=[1, 3, 5], axis=-1)
        self.s = self.add_weight(name="s", shape=(self.n_features, 3), dtype=tf.float32, initializer="glorot_uniform", trainable=True)
        self.s = tf.repeat(self.s, repeats=[1, 3, 5], axis=-1)


    def call(self, inputs):
        # shape (batch, mo, features, angular momentum)
        mo_features = inputs
        tens_prod_1 = self.tp_layer(self.s1[None, None] * mo_features, self.s2[None, None] * mo_features)
        tens_prod_2 = self.tp_layer(self.s2[None, None] * mo_features, self.s1[None, None] * mo_features)
        tens_prod_sum = tens_prod_1 + tens_prod_2
        mo_features = tens_prod_sum + self.k * mo_features
        return mo_features