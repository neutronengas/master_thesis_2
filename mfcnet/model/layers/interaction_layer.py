from tensorflow.keras import layers
import tensorflow as tf

from ..layers.pairmix_layer import PairmixLayer
from ..layers.residual_layer import ResidualLayer
from ..layers.hermite_layer import HermiteLayer
from ..layers.mixing_op import MixingLayer
from ..layers.mixing_op_as import MixingAsLayer

class InteractionLayer(layers.Layer):
    def __init__(self, n_features, n_orb, n_ang_mom, lmax, n_mlp, k, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.n_orb = n_orb

        self.hermite_layer = HermiteLayer(n_features)

        self.pairmix_layer_1 = PairmixLayer(n_features, n_ang_mom)
        self.pairmix_layer_2 = PairmixLayer(n_features, n_ang_mom)
        self.mixing_layer = MixingLayer(n_features, lmax, n_mlp, k)
        self.mixing_as_layer = MixingAsLayer(n_features, lmax, n_mlp, k)
#        self.pairmix_layer_3 = PairmixLayer(n_features)
#        self.pairmix_layer_4 = PairmixLayer(n_features)


  #      self.residual_layer_i = ResidualLayer(F, L, cgc)
  #      self.activation_layer_i = ActivationLayer()
  #      self.spherical_linear_layer_i = SphericalLinearLayer(F, F, L, L, cgc)
#
  #      self.residual_layer_j = ResidualLayer(F, L, cgc)
  #      self.activation_layer_j = ActivationLayer()
  #      self.spherical_linear_layer_j = SphericalLinearLayer(F, F, L, L, cgc)        

 #   def build(self, shape):
#        self.weight_matrices = self.add_weight(name="weight_mtcs", shape=(self.L + 1, self.F,# self.K), dtype=tf.float32, initializer=tf.initializers.GlorotNormal())

    def call(self, inputs):
        
        mo_features, coupling_strengths, mo_neighbours_i, mo_neighbours_j = inputs
        mo_features_mixed = self.mixing_as_layer(mo_features, mo_features)
        mo_i = tf.gather_nd(mo_features_mixed, mo_neighbours_i)
        mo_j = tf.gather_nd(mo_features, mo_neighbours_j)

        #msg = self.pairmix_layer_2((self.pairmix_layer_1((mo_i, mo_j, coupling_strengths)), mo_j, coupling_strengths))
        msg = coupling_strengths * self.mixing_layer(mo_i, mo_j)
        mo_i += msg
        mo_i = tf.math.unsorted_segment_mean(mo_i, mo_neighbours_i, num_segments=len(mo_i))

        mo_i = tf.reshape(mo_i, tf.shape(mo_features))
        return mo_features