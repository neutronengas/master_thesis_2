import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime as dt

def spatial_multipliers_retry(R, coords):

    n_atoms = R.shape[0]

    # R has shape (n_atoms, 3); coords has shape (n_atoms, n_coords, 3), or (n_coords, 3)
    if len(coords.shape) == 2:
        coords = np.repeat(coords[None, :], repeats=len(R), axis=0)
    n_coords = coords.shape[1]
    
    # R_minus_coords has shape (n_atoms, n_coords, 3)
    R_minus_coords = R[:, None, :] - coords

    # transpose the spatial dimension to be the tensor's first, to ease assignment
    R_min_coords_tranposed = np.transpose(R_minus_coords, (2, 0, 1))

    [x, y, z] = R_min_coords_tranposed

    [xy, yz, threez2_minus_r2, xz, x2_minus_y2] = [
        x * y,
        y * z,
        2 * z ** 2 - x ** 2 - y ** 2,
        x * z,
        x ** 2 - y ** 2
    ]

    s_multipliers = np.ones((n_atoms, n_coords, 3))
    p_multipliers = np.concatenate([R_minus_coords, R_minus_coords], axis=2)
    d_multipliers = np.array([xy, yz, threez2_minus_r2, xz, x2_minus_y2]).transpose((1, 2, 0))
    multipliers = np.concatenate([s_multipliers, p_multipliers, d_multipliers], axis=2)
    return multipliers

def create_coeffs_tensor_retry(num_elems=10, json_file="./cc-pvdz.1.json", max_coeff_number=9):
    orig_data = open(json_file).read()
    orig_data = json.loads(orig_data)
    coeffs_per_atom = []
    exponents_per_atom = []
    for z in range(1, num_elems + 1):
        data = orig_data['elements']
        data = data[str(z)]['electron_shells']
        map_func = lambda x: "s" if x == 0 else "p" if x == 1 else "d"
        data = {map_func(ang_mom_data["angular_momentum"][0]): {"exponents": ang_mom_data["exponents"], "coefficients": ang_mom_data["coefficients"]} for ang_mom_data in data}
        s_coefficients = np.array(data["s"]["coefficients"], dtype=np.float32)
        s_coefficients = np.pad(s_coefficients, [(0, 3 - s_coefficients.shape[0]), (0, max_coeff_number - s_coefficients.shape[1])], constant_values=0)

        p_coefficients = np.array(data["p"]["coefficients"], dtype=np.float32)
        p_coefficients = np.repeat(p_coefficients, repeats=3, axis=0)
        p_coefficients = np.pad(p_coefficients, [(0, 6 - p_coefficients.shape[0]), (0, max_coeff_number - p_coefficients.shape[1])], constant_values=0)

        d_coefficients = np.array(data["d"]["coefficients"], dtype=np.float32) if "d" in data.keys() else np.empty(shape=(1, 1))
        d_coefficients = np.repeat(d_coefficients, 5, axis=0)
        d_coefficients = np.pad(d_coefficients, [(0, 0), (0, max_coeff_number - d_coefficients.shape[1])], constant_values=0)


        # order of orbitals is:
        # 1s, 2s, 3s
        # 1px, 1py, 1pz, 2px, 2py, 2pz
        # 3dxy, 3dyz, 3d(3z^2-r^2), 3dxz, 3d(x^2-y^2)
        all_coefficients = np.concatenate([s_coefficients, p_coefficients, d_coefficients])
        coeffs_per_atom.append(all_coefficients)

        s_exponents = np.array(data["s"]["exponents"], dtype=np.float32)[None, :].repeat(len(s_coefficients), axis=0)
        s_exponents = np.pad(s_exponents, [(0, 3 - s_exponents.shape[0]), (0, max_coeff_number - s_exponents.shape[1])], constant_values=1)

        p_exponents = np.array(data["p"]["exponents"], dtype=np.float32)
        p_exponents = p_exponents[None, :].repeat(len(p_coefficients), axis=0)
        p_exponents = np.pad(p_exponents, [(0, 6 - p_exponents.shape[0]), (0, max_coeff_number - p_exponents.shape[1])], constant_values=1)

        d_exponents = np.array(data["d"]["exponents"], dtype=np.float32)[None, :].repeat(len(d_coefficients), axis=0) if "d" in data.keys() else 1 + np.empty(shape=(5, 1))
        d_exponents = np.pad(d_exponents, [(0, 0), (0, max_coeff_number - d_exponents.shape[1])], constant_values=1)
        
        all_exponents = np.concatenate([s_exponents, p_exponents, d_exponents])
        exponents_per_atom.append(all_exponents)


    coeffs_per_atom = np.stack(coeffs_per_atom)
    exponents_per_atom = np.stack(exponents_per_atom)
    return coeffs_per_atom, exponents_per_atom

def create_coeffs_tensor(num_elems=14, json_file="./model/utils/cc-pvdz.1.json"):
    # relative path from location of executing notebook
    
    # test mode
    # data = open("./cc-pvdz.1.json").read()
    # use mode
    orig_data = open(json_file).read()
    
    orig_data = json.loads(orig_data)
    # curtail unnecessary content
    out_tensors = []
    for z in range(1, num_elems + 1):
        # Extract the content inside the single quotes
        data = orig_data['elements']
        data = data[str(z)]['electron_shells']

        # number of angular momenta
        n_angmom = len(data)
        all_coefficients = []
        all_exponents = []
        for i in range(n_angmom):
            
            # number of orbitals per angular momentum: 1 for s, 3 for p, 5 for d
            # the spatial orbital multiplication factors are x, y, z for p and xy, xz, yz, x^2 - y^2, 3z^2 - r^2
            num_orbs = 2 * i + 1
            coefficients = data[i]['coefficients']
            # convert strings to floats
            coefficients = [list(map(float, coeffs)) for coeffs in coefficients]
            coefficients = np.array(coefficients, dtype=np.float32)
            exponents = data[i]['exponents']

            # convert strings to floats
            exponents = [float(exps) for exps in exponents]
            exponents = np.array(exponents, dtype=np.float32)

            # repeat exponents for each set of coefficients
            exponents = np.repeat(exponents[None, :], coefficients.shape[0], axis=0)
            # repeat coefficients / exponents based on number of orbitals per main quantum number: 1 for s, 3 for p, 5 for d
            coefficients = coefficients[:, None]
            coefficients = np.repeat(coefficients, num_orbs, axis=1)

            exponents = exponents[:, None]
            exponents = np.repeat(exponents, num_orbs, axis=1)

            all_coefficients.append(coefficients.tolist())
            all_exponents.append(exponents.tolist())
        
        tens_data = [all_coefficients, all_exponents]
        out = tf.ragged.constant(tens_data)
        out_tensors.append(out)
    out = tf.stack(out_tensors)
    out = out.to_tensor()[:, None]
    return out


def extract_coeffs(Z, m_max, max_no_orbtials_per_m, max_split_per_m, max_coeff_per_ao):
    #ragged_tensor = tf.map_fn((extract_coeffs_elementwise), Z, fn_output_signature=tf.RaggedTensorSpec(shape=[2, None, None, None, None], dtype=tf.float32))
    coeffs_tensor = create_coeffs_tensor()
    out = tf.gather(coeffs_tensor, Z)
    # out = ragged_tensor.to_tensor()[:, None]
    new_dims = tf.concat([out.shape[:3], tf.constant([m_max, max_no_orbtials_per_m, max_split_per_m, max_coeff_per_ao])], axis=0)
    padding = new_dims - out.shape
    padding = tf.stack([tf.zeros(padding.shape, dtype=tf.int32), padding])
    padding = tf.transpose(padding)
    out = tf.pad(out, padding)
    # out has shape (n_atoms, n_coords, 2, m_max=3, max_no_orbitals_per_m=4, max_split_per_m=5, max_coeff_per_ao=12)
    return out