from tensorflow.keras import layers
import tensorflow as tf

from ..outdated_layers.bernstein_layer import BernsteinLayer
from ..layers.tensor_product_contraction import TensorProductContractionLayer
from ..layers.hermite_layer import HermiteLayer


class PairmixLayer(layers.Layer):
    def __init__(self, n_features, n_ang_mom, name='pairmix', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.initializer = tf.keras.initializers.GlorotNormal()

        self.tp_layer = TensorProductContractionLayer(n_ang_mom=n_ang_mom)

        self.hermite_layer = HermiteLayer(n_features=n_features)


    def build(self, shape):
        self.w1 = self.add_weight(name="w1", shape=(self.n_features, self.n_features), initializer=self.initializer)
        self.w2 = self.add_weight(name="w2", shape=(self.n_features, self.n_features), initializer=self.initializer)
                

    def call(self, inputs):
        x, y, g = inputs
        g = self.hermite_layer(g)
        w1g = tf.einsum("fk,nk->nf", self.w1, g)
        w2g = tf.einsum("fk,nk->nf", self.w2, g)
        res = self.tp_layer((w1g[None, None] * x, w2g[None, None] * y)) 
        res += self.tp_layer((w1g[None, None] * y, w2g[None, None] * x))
        return res