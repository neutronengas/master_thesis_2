import numpy as np
import tensorflow as tf
from ..utils.create_orbital_values import create_orbital_values
from datetime import datetime as dt
from ..outdated_layers.tensor_product_expansion import TensorProductExpansionLayer

from tensorflow.keras import layers
from ..layers.pairmix_layer import PairmixLayer

class OutputLayer(layers.Layer):
    def __init__(self, L, F, K, r_cut, cgc, atoms, mode, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.F = F
        self.L = L
        self.atoms = atoms
        self.cgc = cgc
        self.mode = mode
        self.initializer = tf.keras.initializers.GlorotNormal()

        self.pairmix_layer = PairmixLayer(L, L, L, F, K, r_cut, cgc)
        self.tens_prod_exp = TensorProductExpansionLayer(cgc)

    def build(self, shape):
        #self.reduce_feature_vec_matrix = self.add_weight(name="reduce_feature_vec_matrix", shape=(14, self.F,), dtype=tf.float32, trainable=True)
        self.w = self.add_weight(name="weight", shape=(self.F, self.F), initializer=self.initializer)       
        pass

    def call(self, inputs):
        # out: (n_atoms, self.no_orbitals_per_atom, self.emb_size); Z: (n_atoms,); R: (n_atoms, 3); coords: (n_molecule, self.num_grid_points, 3), N: (n_molecule,)
        # atom_pair_indices: (n_pairs, 2), atom_pair_mol_id: (n_pairs,), rdm: (TODO), N_rdm: (TODO)
        molec_feature_vectures, mo_neighbour_indices, mo_mol_id, mo_pair_id, V_n, wfn_pairs, wfn_pairs_mol_id, n_output = inputs

        wfn_pairs_V_n, wfn_pairs_C_n = wfn_pairs
        wfn_pairs_V_n_mol_id, wfn_pairs_V_n_mol_id = wfn_pairs_mol_id

        molec_feature_vectures_i = tf.gather(molec_feature_vectures, mo_neighbour_indices[:0])
        molec_feature_vectures_j = tf.gather(molec_feature_vectures, mo_neighbour_indices[:1])


        if self.mode == "pes":
            molec_pair_scalars = tf.einsum("ni,ij,nj->n", molec_feature_vectures_i, self.w + tf.transpose(self.w), molec_feature_vectures_j)
            res = tf.math.unsorted_segment_sum(molec_pair_scalars, mo_neighbour_indices, n_output)
            return res


        if self.mode == "wfn":
            # wfn_pairs must not contain double pairs
            molec_feature_pairs = tf.gather_nd(molec_feature_vectures, wfn_pairs)
            molec_pair_scalars = tf.einsum("ni,ij,nj->n", molec_feature_pairs[:0], self.w - tf.transpose(self.w), molec_feature_pairs[:1])
            wfn_ampl = tf.math.unsorted_segment_prod(molec_pair_scalars, wfn_pairs_mol_id, n_output)
            return wfn_ampl