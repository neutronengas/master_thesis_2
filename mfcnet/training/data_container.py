import numpy as np
from functools import partial
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag
import os
from scipy.special import binom, factorial
from .data_utils import *


class DataContainer:
    def __init__(self, filename, target, cutoff):
        # 20 molecular orbitals, each consisting of 10 atomic orbitals
        self.cutoff = cutoff
        data_dict = np.load(filename, allow_pickle=True)
        data_dict = data_dict["arr_0"].tolist()

        # keys: R_ao: (None, 20, 4, 3), C: (None, 20, 4, 5), mo_neighbours: (None, 20, 20),
        # h1: (None, ), h2: (None,), h1_idx: (None,), h2_idx: (None,), N_h1: (None,), N_h2: (None,)
        for key in list(data_dict.keys()):
            print(key)
            setattr(self, key, np.array(data_dict[key]))

        self.id = np.arange(len(self.R))
        #self.R = self.R.reshape((-1, self.R.shape[-1]))
        #self.target = data_dict[target[0]]

    def yield_V_n(self):
        return self.V_n

    def __len__(self):
        return self.id.shape[0]
    
    def _bmat_fast(self, mats):
        new_data = np.concatenate([mat.data for mat in mats])

        ind_offset = np.zeros(1 + len(mats))
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))])

        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1:] + indptr_offset[i] for i in range(len(mats))])
        return sp.csr_matrix((new_data, new_indices, new_indptr))

    
    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]
        data = {}
        data["target"] = np.array([[0.]])
        data["idx"] = self.id[idx]

        # data["target"] = self.target[idx]

        data['Z'] = self.Z[idx]
        data['R'] = self.R[idx]
        data['C'] = self.C[idx]
        data['S'] = self.S[idx]

#        data['N_h1'] = self.N_h1[idx]
#        data['N_h2'] = self.N_h2[idx]
#
#        data['h1'] = np.zeros(np.sum(data['N_h1']), dtype=np.int32)
#        data['h2'] = np.zeros(np.sum(data['N_h2']), dtype=np.int32)
#        data['h1_idx'] = np.zeros(np.sum(data['N_h1']), dtype=np.int32)
#        data['h2_idx'] = np.zeros(np.sum(data['N_h2']), dtype=np.int32)
#
#        #data['target'] = np.zeros((np.sum(data["N"])), dtype=np.float32)
#            
#        n_h1_end = 0
#        n_h2_end = 0
#        adj_matrices = []
#        for k, i in enumerate(idx):
#
#            n_h1 = data['N_1'][k]
#            n_h2 = data['N_2'][k]
#
#
#            n_h1_start = n_h1_end
#            n_h1_end = n_h1_start + n_h1
#            n_h2_start = n_h2_end
#            n_h2_end = n_h2_start + n_h2
#
#
#            h1 = self.h1[self.N_h1_cumsum[i]:self.N_h1_cumsum[i + 1]]
#            data['h1'][n_h1_start:n_h1_end] = h1
#
#            h2 = self.h2[self.N_h2_cumsum[i]:self.N_h2_cumsum[i + 1]]
#            data['h2'][n_h2_start:n_h2_end] = h2
#
#            h1_idx = self.h1[self.N_h1_cumsum[i]:self.N_h1_cumsum[i + 1]]
#            data['h1_idx'][n_h1_start:n_h1_end] = h1_idx
#
#            h2_idx = self.h1[self.N_h2_cumsum[i]:self.N_h2_cumsum[i + 1]]
#            data['h2_idx'][n_h2_start:n_h1_end] = h2_idx
#
#
        data['h1'] = self.h1[idx]
        data['h1_idx'] = self.h1_idx[idx]

        data['h2'] = self.h2[idx]
        data['h2_idx'] = self.h2_idx[idx]

        data['V_n'] = self.V_n[idx]

        mo_neighbours = self.mo_neighbours[idx]
        nnz = np.nonzero(mo_neighbours)
        t, i, j = nnz
        mo_neighbours_i = np.stack([t, i])
        mo_neighbours_i = np.ravel_multi_index(mo_neighbours_i, dims=(len(idx), 20))
        mo_neighbours_j = np.stack([t, j])
        mo_neighbours_j = np.ravel_multi_index(mo_neighbours_j, dims=(len(idx), 20))
        data['mo_neighbours_i'] = mo_neighbours_i
        data['mo_neighbours_j'] = mo_neighbours_j
        data['coupl'] = mo_neighbours[nnz]
        n_orb = len(mo_neighbours[0])
        data["gram"] = mo_neighbours * (np.ones(n_orb, n_orb) - np.eye(n_orb))[None, :, :]
        return data