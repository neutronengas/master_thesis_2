import numpy as np
import tensorflow as tf
from ..utils.create_orbital_values import create_orbital_values
from datetime import datetime as dt

from tensorflow.keras import layers

class OutputBlock(layers.Layer):
    def __init__(self, emb_size, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size

    def build(self, shape):
        self.reduce_feature_vec_matrix = self.add_weight(name="reduce_feature_vec_matrix", shape=(14, self.emb_size,), dtype=tf.float32, trainable=True)

    def call(self, inputs):
        # out: (n_atoms, self.no_orbitals_per_atom, self.emb_size); Z: (n_atoms,); R: (n_atoms, 3); coords: (n_molecule, self.num_grid_points, 3), N: (n_molecule,)
        # atom_pair_indices: (n_pairs, 2), atom_pair_mol_id: (n_pairs,), rdm: (TODO), N_rdm: (TODO)
        out, Z, R, N, atom_pair_indices, atom_pair_mol_id, rdm, N_rdm = inputs

        # reshape out to DM form
        # after this step, out has shape (n_pairs, 2, self.no_orbitals_per_atom, self.emb_size)
        out = tf.gather(out, atom_pair_indices)
        
        # multiply with step-variable to make sure the padded elements of the rdm belonging to H / He still are zero
        step_var = tf.constant([1 for _ in range(5)] + [0 for _ in range(9)], dtype=tf.float32)
        multiplier = tf.stack([step_var, step_var] + [tf.ones((14,)) for _ in range(12)], axis=0)
        multiplier = tf.gather(multiplier, tf.gather(Z, atom_pair_indices))[:, :, :, None]
        out = out * multiplier

        # after this step, out has shape (n_pairs, self.no_orbitals_per_atom, self.no_orbitals_per_atom, self.emb_size) and is symmetric wrt to the to orbital dimensions)
        #out = out[:, 0][:, None, :, :] * out[:, 1][:, :, None, :] + out[:, 0][:, None, :, :] + out[:, 1][:, :, None, :]
        out = out[:, 0][:, None, :, :] + out[:, 1][:, :, None, :]


        # reduce feature vector to single number
        out = tf.einsum("nijk,ijk->nij", out, self.reduce_feature_vec_matrix[None, :, :] * self.reduce_feature_vec_matrix[:, None, :])   

        # number of molecules within batch
        n_mol = tf.shape(N)[0]

        # create all evaluated orbitals for all the atoms
        # orbitals: (n_atoms, n_coords, 14)
        #orbitals = tf.numpy_function(create_orbital_values, [Z, R, coords], Tout=tf.float32)

        ## grab the respective orbitals for each atom pair
        #orbitals = tf.gather(orbitals, atom_pair_indices)

        ## outer product of the orbitals of each atom pair
        #orbitals = orbitals[:, 0, :, :, None] * orbitals[:, 1, :, None]

        # add correction term to the rdm
        rdm += out

        # multiply orbitals with corresponding segments of the rdm
        # rho = tf.reduce_sum(orbitals * rdm[:, None], axis=(-1, -2))
        
        # sum over the atom-pairs, grouped by the molecule to which they belong
        # output shape: (n_mol, n_coords)
        # densities_molecule_wise = tf.math.unsorted_segment_sum(rho, atom_pair_mol_id, num_segments=n_mol)
        
        return rdm