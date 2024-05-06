from tensorflow.keras import layers

from ..outdated_layers.spherical_linear_layer import SphericalLinearLayer
from .activation_layer import ActivationLayer

class ResidualLayer(layers.Layer):
    def __init__(self, F, L, cgc, name='residual', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.spherical_linear_layer1 = SphericalLinearLayer(F, F, L, L, cgc)
        self.spherical_linear_layer2 = SphericalLinearLayer(F, F, L, L, cgc)
        
        self.activation1 = ActivationLayer()
        self.activation2 = ActivationLayer()
        
    def call(self, inputs):
        res = self.activation1(res)
        res = self.spherical_linear_layer1(inputs)
        res = self.activation2(res)
        res = self.spherical_linear_layer2(res)
        res = [res[i] + inputs[i] for i in range(len(res))]
        return res