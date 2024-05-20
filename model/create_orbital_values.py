import numpy as np
import json

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


def s_norms(exponents, coefficients):
    # exponents: (n_atoms, 1, n_orbitals=3, n_gtos=9), coefficients: (n_atoms, 1, n_orbitals=3, n_gtos=9)
    norm = coefficients[:, :, :, :, None] * coefficients[:, :, :, None, :]
    norm *= (np.pi / (exponents[:, :, :, :, None] + exponents[:, :, :, None, :])) ** 1.5
    norm = norm.sum(axis=(-1, -2))[:, :, :, None] ** 0.5
    return norm

def p_norms(exponents, coefficients):
    # exponents: (n_atoms, 1, n_orbitals=6, n_gtos=9), coefficients: (n_atoms, 1, n_orbitals=6, n_gtos=9)
    norm = coefficients[:, :, :, :, None] * coefficients[:, :, :, None, :]
    #print(coefficients[0, 0, 0])
    norm *= np.pi ** 1.5 / (2 * (exponents[:, :, :, :, None] + exponents[:, :, :, None, :]) ** 2.5) 
    norm = norm.sum(axis=(-1, -2))[:, :, :, None] ** 0.5
    norm[norm == 0.0] = 1.
    #norm[:, :, :3] /= norm[:, :, 3:]
    norm[:, :, 3:] /= norm[:, :, 3:]
    return norm

def d_norms(exponents, coefficients):
    # exponents: (n_atoms, 1, n_orbitals=5, n_gtos=9), coefficients: (n_atoms, 1, n_orbitals=6, n_gtos=9)
    # 3dxy, 3dyz, 3d(3z^2-r^2), 3dxz, 3d(x^2-y^2)
    norm = coefficients[:, :, :, :, None] * coefficients[:, :, :, None, :]
    exponents = np.concatenate([
        [np.pi ** 1.5 / (4 * (exponents[:, :, 0, :, None] + exponents[:, :, 0, None, :]) ** 3.5)],
        [np.pi ** 1.5 / (4 * (exponents[:, :, 1, :, None] + exponents[:, :, 1, None, :]) ** 3.5)],
        [12 * np.pi ** 1.5 / (4 * (exponents[:, :, 2, :, None] + exponents[:, :, 2, None, :]) ** 3.5)],
        [np.pi ** 1.5 / (4 * (exponents[:, :, 3, :, None] + exponents[:, :, 3, None, :]) ** 3.5)],
        [4 * np.pi ** 1.5 / (4 * (exponents[:, :, 4, :, None] + exponents[:, :, 4, None, :])) ** 3.5]
    ], axis=0)
    exponents = exponents.transpose((1, 2, 0, 3, 4))
    norm *= exponents
    norm = norm.sum(axis=(-1, -2))[:, :, :, None] ** 0.5
    return norm

def norms(exponents, coefficients):
    s_norms_eval = s_norms(exponents[:, :, :3, :], coefficients[:, :, :3, :])
    p_norms_eval = p_norms(exponents[:, :, 3:9, :], coefficients[:, :, 3:9, :])
    d_norms_eval = d_norms(exponents[:, :, 9:, :], coefficients[:, :, 9:, :])
    norms = np.concatenate([s_norms_eval, p_norms_eval, d_norms_eval], axis=2)
    norms[norms == 0.] = 1.
    return norms

def create_coeffs_tensor_retry(num_elems=10, json_file="./utils/cc-pvdz.1.json", max_coeff_number=9):
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

def create_prefactors(exponents):
    # orbitals are (1s, 2s, 3s, 2px, 2py, 2pz, 3px, 3py, 3pz, 3d(xy), 3d(yz), 3d(z2-r2), 3d(xz), 3d(x2-y2))
    i = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 1, 2], dtype=np.int32)[:, None]
    j = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 2, 0, 2], dtype=np.int32)[:, None]
    k = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 1, 0])[:, None]
    fac = np.vectorize(np.math.factorial)
    prefactors = (2 * exponents / np.pi) ** 0.75
    prefactors *= ((8 * exponents) ** (i + j + k)) ** 0.5
    prefactors *= (fac(i) * fac(j) * fac(k) / (fac(2 * i) * fac(2 * j) * fac(2 * k))) ** 0.5
    prefactors *= (-1) ** (i + j + k)
    return prefactors

def create_orbital_values(Z, R, coords):
    R_minus_coords = np.linalg.norm(R[:, None] - coords, axis=-1)[:, :, None, None] ** 2
    coeffs_orig, exponents = create_coeffs_tensor_retry()
    coeffs_orig = coeffs_orig[Z - 1][:, None]
    exponents = exponents[Z - 1][:, None]
    spatial_multipliers = spatial_multipliers_retry(R, coords)[:, :, :, None]
    prefactors = create_prefactors(exponents)
    coeffs = coeffs_orig * spatial_multipliers * prefactors
    norms_eval = norms(exponents, coeffs_orig * prefactors)
    #norms_eval = 1
    orbital_values = (coeffs / norms_eval * np.exp(-1 * exponents * R_minus_coords)).sum(axis=-1)
    return orbital_values.astype(np.float32)