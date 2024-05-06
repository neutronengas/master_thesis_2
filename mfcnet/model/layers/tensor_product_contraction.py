from tensorflow.keras import layers
import tensorflow as tf

from ..utils.cgc_multiplier import cgc_multiplier

class TensorProductContractionLayer(layers.Layer):
    def __init__(self, n_ang_mom, name='activation', **kwargs):
        super().__init__(name=name, **kwargs)

        self.cgc = tf.convert_to_tensor(cgc_multiplier(n_ang_mom))

    def call(self, inputs):
        # shapes (batch, molecular orbitals, features, angular momentum)
        x, y = inputs
        res = tf.einsum("bmaf,bmcf,adc->bmdf", x, y, self.cgc)
        return res