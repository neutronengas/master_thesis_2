import numpy as np
import tensorflow as tf
from ..utils.create_orbital_values import create_orbital_values
from datetime import datetime as dt
from .tensor_product_expansion import TensorProductExpansionLayer

from tensorflow.keras import layers
from ..layers.pairmix_layer import PairmixLayer

class OutputLayer(layers.Layer):
    def __init__(self, L, F, K, r_cut, cgc, atoms, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.F = F
        self.L = L
        self.atoms = atoms
        self.initializer = tf.keras.initializers.GlorotNormal()

        self.pairmix_layer = PairmixLayer(L, L, L, F, K, r_cut, cgc)
        self.tens_prod_exp = TensorProductExpansionLayer(cgc)

    def build(self, shape):
        #self.reduce_feature_vec_matrix = self.add_weight(name="reduce_feature_vec_matrix", shape=(14, self.F,), dtype=tf.float32, trainable=True)
        self.weight = self.add_weight(name="weight", shape=(len(self.atoms) ** 2, self.L, self.F, self.F), initializer=self.initializer)       
        self.s_collapse = self.add_weight(name="s_collapse", shape=(3, self.F), initializer=self.initializer)
        self.p_collapse = self.add_weight(name="p_collapse", shape=(2, self.F), initializer=self.initializer)
        self.d_collapse = self.add_weight(name="d_collapse", shape=(1, self.F), initializer=self.initializer)
        self.collapses = [self.s_collapse, self.p_collapse, self.d_collapse]

    def call(self, inputs):
        # out: (n_atoms, self.no_orbitals_per_atom, self.emb_size); Z: (n_atoms,); R: (n_atoms, 3); coords: (n_molecule, self.num_grid_points, 3), N: (n_molecule,)
        # atom_pair_indices: (n_pairs, 2), atom_pair_mol_id: (n_pairs,), rdm: (TODO), N_rdm: (TODO)
        c, Z, R, N, atom_pair_indices, atom_pair_mol_id, atom_idx, rdm, N_rdm = inputs
        c_pairs = [tf.gather(c[l], atom_pair_indices) for l in range(len(c))]
        R_ij = tf.gather(R, atom_pair_indices)
        R_ij = R_ij[:, 0, :] - R_ij[:, 1, :]
        x = [x[:, 0] for x in c_pairs]
        y = [y[:, 1] for y in c_pairs]
        # s * s -> s: (1s, 1s), (1s, 2s), (1s, 3s), (2s, 2s), (2s, 3s), (3s, 3s) (6 pairs)
        # s * p -> p: (1s, 2p), (1s, 3p), (2s, 2p), (2s, 3p), (3s, 2p), (3s, 3p) (6 pairs)
        # s * d -> d: (1s, 3d), (2s, 3d), (3s, 3d) (3 pairs)
        # p * p -> s + p + d: (2p, 2p), (2p, 3p), (3p, 3p) (3 pairs)
        # p * d -> s + p + d + f: (2p, 3d), (3p, 3d) (2 pairs)

        c_out = self.pairmix_layer([x, y, R_ij])
        #tf.print(tf.reduce_max(c_out[0]))
        atom_idx = tf.gather(atom_pair_indices, atom_idx)
        expanded_idx_1 = atom_idx[:, 0] * len(self.atoms) + atom_idx[:, 1]
        expanded_idx_2 = atom_idx[:, 1] * len(self.atoms) + atom_idx[:, 0]
        gathered_weights = tf.gather(self.weight, expanded_idx_1) + tf.gather(self.weight, expanded_idx_2)
        for l in range(self.L):
            weights_rel = gathered_weights[l, :, :]
            c_out[l] = tf.einsum("abc,icd->ibd", weights_rel, c_out[l])
        M = {}
        for l1 in range(3):
            for l2 in range(l1 + 1):
                M_l1l2 = tf.zeros(shape=(len(c_out[0]), 2*l1+1, 2*l2+1, self.F))
                l3_range = [l1 - x for x in range(-l2, l2 + 1)]
                for l3 in l3_range:
                    self.tens_prod_exp.set_params(l1, l2, l3)
                    M_l1l2 += self.tens_prod_exp([c_out])
                    #tf.print(tf.reduce_max(M_l1l2))
                M[(l1, l2)] = M_l1l2
        for (l1, l2) in M:
            l1_collapse = self.collapses[l1]
            l2_collapse = self.collapses[l2]
            M[(l1, l2)] = tf.einsum("imnf,af,bf->imnab", M[(l1, l2)], l1_collapse, l2_collapse)
            M[(l1, l2)] = tf.reshape(M[(l1, l2)], shape=(len(M[(l1, l2)]), M[(l1, l2)].shape[1] * M[(l1, l2)].shape[3], M[(l1, l2)].shape[2] * M[(l1, l2)].shape[4]))
            
        # s-s, p-s, d-s
        rdm_1 = tf.concat([M[(0, 0)], M[(1, 0)], M[(2, 0)]], axis=1)
        # s-p, p-p, d-p
        rdm_2 = tf.concat([tf.transpose(M[(1, 0)], (0, 2, 1)), M[(1, 1)], M[(2, 1)]], axis=1)
        # s-d, p-d, d-d
        rdm_3 = tf.concat([tf.transpose(M[(2, 0)], (0, 2, 1)), tf.transpose(M[(2, 1)], (0, 2, 1)), M[(2, 2)]], axis=1)
        # concatenating
        rdm = tf.concat([rdm_1, rdm_2, rdm_3], axis=2)
        return rdm