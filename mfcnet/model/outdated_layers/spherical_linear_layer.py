from tensorflow.keras import layers
import tensorflow as tf

from ..layers.linear_layer import LinearLayer
from ..layers.selfmix_layer import SelfmixLayer

class SphericalLinearLayer(layers.Layer):
    def __init__(self, Fin, Fout, Lin, Lout, cgc, name='selfmix', **kwargs):
        super().__init__(name=name, **kwargs)

        self.linear_layer = LinearLayer(Fin, Fout)
        self.selfmix_layer = SelfmixLayer(cgc, Lin, Lout, Fin)

    def call(self, inputs):
        res = self.selfmix_layer(self.linear_layer(inputs))
        return res