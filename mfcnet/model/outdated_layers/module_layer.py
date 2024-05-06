from tensorflow.keras import layers
import tensorflow as tf

from .spherical_linear_layer import SphericalLinearLayer
from ..layers.pairmix_layer import PairmixLayer
from ..layers.sph_harm_layer import SphHarmLayerl
from .bernstein_layer import BernsteinLayer

class ModuleLayer(layers.Layer):
    def __init__(self, L, F, K, r_cut, cgc, name='module', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.L = L
        self.F = F
        self.K = K
        self.spherical_linear_layer_a = SphericalLinearLayer(1, F, L, L, cgc)
        self.spherical_linear_layer_b = SphericalLinearLayer(1, F, L, L, cgc)
        self.pairmix_layer_1 = PairmixLayer(L, L, L, F, K, r_cut, cgc)
        self.pairmix_layer_2 = PairmixLayer(L, L, L, F, K, r_cut, cgc)
        self.bernstein_layer = BernsteinLayer(K, r_cut)
        self.sph_harm_layers = [SphHarmLayerl(l) for l in range(L + 1)]

    def build(self, shape):
        self.weight_matrices = self.add_weight(name="weight_mtcs", shape=(self.L + 1, self.F, self.K), dtype=tf.float32, initializer=tf.initializers.GlorotNormal())

    def call(self, inputs):
        # c: (None, F, (L + 1)^2)
        atoms, pairs, atom_pair_indices, R, Y = inputs
        R_ij = tf.gather(R, atom_pair_indices[:, 0]) - tf.gather(R, atom_pair_indices[:, 1])
        atoms_i = [tf.gather(atoms[i], atom_pair_indices[:, 0]) for i in range(self.L + 1)]
        atoms_j = [tf.gather(atoms[i], atom_pair_indices[:, 1]) for i in range(self.L + 1)]

        nj = [tf.gather(atoms[i], atom_pair_indices[:, 1]) for i in range(self.L + 1)]
        nj0 = nj[0]

        Y = [tf.repeat(Y_l[:, None, :], repeats=self.F, axis=1) for Y_l in Y]
        b = self.pairmix_layer_1((nj, Y, tf.norm(R_ij, axis=1)))
        atoms_mixed = self.pairmix_layer_2((atoms_i, atoms_j, tf.norm(R_ij, axis=1)))
        atoms_mixed = [0.5 * x for x in atoms_mixed]

        for l in range(self.L + 1):
            weight_mtx = self.weight_matrices[l]
            g = self.bernstein_layer(tf.norm(R_ij, axis=1))
            a_l = nj0 * tf.einsum("ij,nj->ni", weight_mtx, g)[:, :, None]
            a_l *= Y[l]
            b_l = b[l]
            msg = tf.math.unsorted_segment_sum(a_l + b_l, atom_pair_indices[:, 1], num_segments=len(atoms))
            atoms[l] += msg

            pairs[l] += atoms_mixed[l]
        return atoms, pairs