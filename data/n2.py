# -*- coding: utf-8 -*-

import numpy as np
from pyscf import gto, scf, lo, cc, ao2mo, fci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import block_diag
from itertools import combinations
from scipy.special import binom

norb = 10
nelec = (7,7)

"""**Creating all FCI strings**"""

def all_dets(norb, nelec):
    all_strings = [list(I) for I in combinations(range(norb), nelec)]
    all_strings = [[y in x for y in range(norb)] for x in all_strings]
    all_strings = [x + y for x in all_strings for y in all_strings]
    return all_strings

def create_perm_matrix(vec, norb):
    if len(vec) > norb:
        u = block_diag(create_perm_matrix(vec[:norb], norb), create_perm_matrix(vec[norb:], norb))
        return u
    nelec = vec.sum()
    holes = np.nonzero(1 - vec[:nelec])[0]
    excited = np.nonzero(vec[nelec:])[0] + nelec
    idx = np.arange(len(vec))
    idx[holes] = excited
    idx[excited] = holes
    u = np.eye(len(vec))[idx]
    return u

def create_neighbour_dets(det, norb, nelec):
    # norb: number of spatial orbitals; nelec: number of electrons for alpha-/beta-orbitals (=nelec_total / 2)
    single_exc_des = [list(c) for c in combinations(range(nelec), nelec - 1)]
    single_exc_des = [[y in x for y in range(nelec)] for x in single_exc_des]
    single_exc_cre = [list(c) for c in combinations(range(norb - nelec), 1)]
    single_exc_cre = [[y in x for y in range(norb - nelec)] for x in single_exc_cre]
    single_exc = [des + cre for des in single_exc_des for cre in single_exc_cre]
    double_exc_des = [list(c) for c in combinations(range(nelec), nelec - 2)]
    double_exc_des = [[y in x for y in range(nelec)] for x in double_exc_des]
    double_exc_cre = [list(c) for c in combinations(range(norb - nelec), 2)]
    double_exc_cre = [[y in x for y in range(norb - nelec)] for x in double_exc_cre]
    double_exc = [des + cre for des in double_exc_des for cre in double_exc_cre]
    hf_state = [i < nelec for i in range(norb)]
    single_exc_sp = [s_exc + hf_state for s_exc in single_exc]
    single_exc_sp += [hf_state + s_exc for s_exc in single_exc]
    double_exc_sp = [d_exc + hf_state for d_exc in double_exc]
    double_exc_sp += [hf_state + d_exc for d_exc in double_exc]
    double_exc_sp += [s_exc_a + s_exc_b for s_exc_a in single_exc for s_exc_b in single_exc]
    all_exc_sp = np.concatenate([np.array(single_exc_sp), np.array(double_exc_sp), np.array([hf_state + hf_state])], axis=0).astype(int)
    u = create_perm_matrix(np.array(det).astype(int), norb)
    all_exc_sp = np.einsum("ab,cb->ca", u, all_exc_sp)
    return all_exc_sp

def my_dets_to_pyscf_addrs(det, norb, nelec):
    det_a = "".join(det[:norb].astype(str))[::-1]
    det_b = "".join(det[norb:].astype(str))[::-1]
    addr_a = fci.cistring.str2addr(norb, nelec, det_a)
    addr_b = fci.cistring.str2addr(norb, nelec, det_b)
    num_str = fci.num_strings(norb, nelec)
    addr = addr_a * num_str + addr_b
    return addr

n_orb = 10
nelec = (7,7)
neighbour_dets = np.stack([create_neighbour_dets(det, norb, nelec[0]) for det in all_dets(norb, nelec[0])], axis=0)

all_dets_bins = np.array(all_dets(norb, nelec[0]), dtype=np.int32)
pyscf_order = np.apply_along_axis(my_dets_to_pyscf_addrs, arr=all_dets_bins.astype(int), axis=-1, norb=norb, nelec=nelec[0])
pyscf_order = np.argsort(pyscf_order)
all_dets_bins_ordered = all_dets_bins[pyscf_order]
all_dets_as_idx = np.apply_along_axis(lambda x: np.nonzero(x)[0], axis=-1, arr=all_dets_bins_ordered)
neighbour_dets = neighbour_dets[pyscf_order]
neighbour_dets_bins = neighbour_dets.copy()
neighbour_dets_calc = (neighbour_dets * (2 ** np.arange(2 * norb))).sum(axis=-1)
neighbour_dets_j = np.apply_along_axis(my_dets_to_pyscf_addrs, arr=neighbour_dets.astype(int), axis=-1, norb=norb, nelec=nelec[0])

"""**Creating 1200 training samples**"""

# 1200 random points, homogeneously distributed over the sphere
points = np.random.normal(size=(1200, 3))
points = points / (2 * np.linalg.norm(points, axis=1)[:, None])
[x, y, z] = points.T

def n2_data_generator(n_samples, points):
  basis = "sto-3g"
  # molecule: N2
  n2_bond_length = 0.55
  n2_min_bond_length = 0.5 * n2_bond_length
  n2_max_bond_length = 2 * n2_bond_length
  n2_bond_lengths = np.linspace(n2_min_bond_length, n2_max_bond_length, num=n_samples)
  n2_points = np.stack([points, -points]).transpose((1, 0, 2))
  n2_atoms = [["N", "N"] for _ in n2_points]

  i = 0
  while i < n_samples:
    n2_coords = n2_bond_lengths[i] * n2_points[i]
    atom_data = list(zip(n2_atoms[i], n2_coords.tolist()))
    mol = gto.Mole(atom=atom_data, basis=basis)
    mol.build()
    S = mol.intor("int1e_ovlp")


    mf = scf.RHF(mol)
    mf.kernel()
    e_hf = mf.e_tot
    #n2_e_hf.append(e_hf)

    #C = lo.orth_ao(mf, 'meta-lowdin')
    C = mf.mo_coeff
    #n2_c_local.append(C.tolist())

    h1 = mf.get_hcore()
    h1 = np.einsum("ba,be,ef->af", C, h1, C)
    h2 = mol.intor("int2e", aosym="s1")
    h2 = np.einsum("ba,xbey,ef->xafy", C, h2, C)
    h2 = np.einsum("ba,bxye,ef->axyf", C, h2, C)

    na = nb = fci.num_strings(norb, nelec[0])
    H_fci = fci.direct_spin1.pspace(h1, h2, norb, nelec, np=na * nb)

    matrix_els = np.fromfunction(lambda i, j: H_fci[1][i, neighbour_dets_j[i, j]], shape=neighbour_dets_j.shape, dtype=np.int32)
    #n2_matr_els.append(matrix_els.tolist())

    mycc = cc.CCSD(mf)
    mycc.kernel()
    e_cc = mycc.e_tot
    #n2_e_cc.append(e_cc)
    yield np.array([e_hf, e_cc]).tolist, S, C, matrix_els, H_fci[1][0, 0]
    i += 1

generator = n2_data_generator(100, points)

i = 0
for e, S, C, m, _ in generator:
  with open(f"data/data_{i}.npz", "wb") as file:
    np.savez(file, e=e, S=S, C=C, m=m)
  i += 1