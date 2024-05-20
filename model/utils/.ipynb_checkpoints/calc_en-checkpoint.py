import tensorflow as tf

def calc_amplitude(x_vec_neighbours_idx, mo_features, gram, norb, cgc):
    sp_a_idx = x_vec_neighbours_idx[:, :norb]
    sp_b_idx = x_vec_neighbours_idx[:, norb:]
    ampl_a = tf.gather(mo_features, sp_a_idx)
    ampl_a_sq = tf.einsum("niaf,nibf,acb->nicf", ampl_a, ampl_a, cgc)
    ampl_b = tf.gather(mo_features, sp_b_idx)
    ampl_b_sq = tf.einsum("niaf,nibf,acb->nicf", ampl_b, ampl_b, cgc)
    ampl_a = tf.linalg.det("niaf,njaf,nij->nij", ampl_a, ampl_a_sq, gram)
    ampl_b = tf.linalg.det("niaf,njaf,nij->nij", ampl_b, ampl_b_sq, gram)
    ampl = ampl_a * ampl_b
    return ampl


def e_local(N_u, x_vec_as_dec, x_vec_as_idx, dets_as_idx, dets_as_dec, neighbour_dets_as_dec, neighbour_dets_matrix_el, mo_features, gram):
    # x_vec_as_dec (= x_vec): (N_u,); x_vec_as_idx: (N_u, nelec); dets_as_idx: (no_dets, nelec)
    # neighbour_dets_as_dec: (no_dets, no_neighbour_dets); neighbour_dets_matrix_el: (no_dets, no_neighbour_dets)
    norb = len(dets_as_idx[0])
    x_vec_neighbours_dec = tf.gather(neighbour_dets_as_dec, x_vec_as_dec)
    x_vec_neighbours_idx = tf.gather(dets_as_idx, x_vec_neighbours_dec)
    ampls = calc_amplitude(x_vec_neighbours_idx, mo_features, gram, norb)
    # ampls of shape (V_n, no_neighbour_dets)
    matr_els = tf.gather(neighbour_dets_matrix_el, x_vec_as_idx)
    denom = calc_amplitude(x_vec_as_idx, mo_features, gram, norb)
    e_loc = tf.reduce_sum(ampls * matr_els, axis=-1) / denom
    top_N_u_ampls_idx = tf.argsort(tf.reshape(ampls, [tf.shape(ampls)[0], tf.shape(ampls)[1] * tf.shape(ampls)[2]]))[:, -N_u:]
    top_N_u_dets_as_idx = tf.gather(dets_as_idx, top_N_u_ampls_idx)
    top_N_u_dets_as_dec = tf.gather(dets_as_dec, top_N_u_ampls_idx)
    return e_loc, denom, top_N_u_dets_as_idx, top_N_u_dets_as_dec

def create_neighbour_states(mo_features, x_vec, h1, h1_idx, h2, h2_idx, n_features, N_u, gram):
    x_vec = tf.cast(x_vec, dtype=tf.int32)
    mp_one = tf.cast(tf.math.floormod(tf.range(n_features), 2) * 2 - 1, dtype=tf.float32)

    # mo_features of shape (mo_orbital, feature channels, angular momentum)
    n_orb = len(x_vec)
    n_e = tf.reduce_sum(x_vec)
    
    h1_states = tf.repeat(x_vec[:, None], repeats=len(h1_idx), axis=1)

    h1_idx_minus = h1_idx[:, 1, None]
    h1_idx_plus = h1_idx[:, 0, None]

    h1_minus_shift = tf.reduce_sum(tf.cast(tf.equal(tf.range(n_orb)[:, None, None], h1_idx_minus), tf.int32), axis=-1)
    h1_states -= h1_minus_shift
    h1_phase_fac = tf.cast(tf.range(n_orb)[:, None, None] < h1_idx_minus, tf.int32)[:, :, 0]
    h1_phase_fac = tf.abs(h1_phase_fac)
    h1_phase_exp = tf.reduce_sum(h1_phase_fac * h1_states, axis=0)
    h1_phase = (-1) ** tf.abs(h1_phase_exp)
    h1_legal = tf.reduce_min(h1_states, axis=0) == 0
    
    h1_plus_shift = tf.reduce_sum(tf.cast(tf.equal(tf.range(n_orb)[:, None, None], h1_idx_plus), tf.int32), axis=-1)
    h1_states += h1_plus_shift
    h1_phase_fac = tf.cast(tf.range(n_orb)[:, None, None] < h1_idx_plus, tf.int32)[:, :, 0]
    h1_phase_fac = tf.abs(h1_phase_fac)
    h1_phase_exp = tf.reduce_sum(h1_phase_fac * h1_states, axis=0)
    h1_phase *= (-1) ** tf.abs(h1_phase_exp)
    h1_legal = tf.logical_and(tf.reduce_max(h1_states, axis=0) == 1, h1_legal)
    
    h1 = h1[h1_legal] * tf.cast(h1_phase, tf.float32)[h1_legal]
    h1_states = tf.boolean_mask(h1_states, h1_legal, axis=1)
    
    h1_phase = h1_phase[h1_legal]
    h1_states_idx = tf.where(h1_states > 0)
    h1_states_idx = tf.gather(h1_states_idx, tf.argsort(h1_states_idx[:, 1]))[:, 0]
    h1_states_idx = tf.reshape(h1_states_idx, (len(h1), n_e))
    h1_ampls = tf.gather(mo_features, h1_states_idx)
    tmp = tf.einsum("lnfa,lmfa,f,lnm->lnm", h1_ampls, h1_ampls ** 2, mp_one, gram)
    h1_ampls = tf.linalg.det(tmp) * tf.cast(h1_phase, dtype=tf.float32)

    h2_states = tf.repeat(x_vec[:, None], repeats=len(h2_idx), axis=1)

    h2_minus_shift = tf.reduce_sum(tf.cast(tf.equal(tf.range(n_orb)[:, None, None], h2_idx[:, 1, None]), tf.int32), axis=-1)
    h2_states -= h2_minus_shift
    h2_legal = tf.reduce_min(h2_states, axis=0) == 0
    h2_phase_fac = tf.cast(tf.range(n_orb)[:, None, None] < h2_idx[:, 1], tf.int32)[:, 0]
    h2_phase_fac = tf.abs(h2_phase_fac)
    h2_phase_exp = tf.reduce_sum(h2_phase_fac * h2_states, axis=0)
    h2_phase = (-1) ** tf.abs(h2_phase_exp)

    h2_minus_shift = tf.reduce_sum(tf.cast(tf.equal(tf.range(n_orb)[:, None, None], h2_idx[:, 3, None]), tf.int32), axis=-1)
    h2_states -= h2_minus_shift
    h2_legal = tf.logical_and(h2_legal, tf.reduce_min(h2_states, axis=0) == 0)
    h2_phase_fac = tf.cast(tf.range(n_orb)[:, None, None] < h2_idx[:, 3], tf.int32)[:, 0]
    h2_phase_fac = tf.abs(h2_phase_fac)
    h2_phase_exp = tf.reduce_sum(h2_phase_fac * h2_states, axis=0)
    h2_phase *= (-1) ** tf.abs(h2_phase_exp)

    h2_plus_shift = tf.reduce_sum(tf.cast(tf.equal(tf.range(n_orb)[:, None, None], h2_idx[:, 2, None]), tf.int32), axis=-1)
    h2_states += h2_plus_shift
    h2_legal = tf.logical_and(h2_legal, tf.reduce_max(h2_states, axis=0) == 1)
    h2_phase_fac = tf.cast(tf.range(n_orb)[:, None, None] < h2_idx[:, 2], tf.int32)[:, 0]
    h2_phase_fac = tf.abs(h2_phase_fac)
    h2_phase_exp = tf.reduce_sum(h2_phase_fac * h2_states, axis=0)
    h2_phase *= (-1) ** tf.abs(h2_phase_exp)

    h2_plus_shift = tf.reduce_sum(tf.cast(tf.equal(tf.range(n_orb)[:, None, None], h2_idx[:, 0, None]), tf.int32), axis=-1)
    h2_states += h2_plus_shift
    h2_legal = tf.logical_and(h2_legal, tf.reduce_max(h2_states, axis=0) == 1)
    h2_phase_fac = tf.cast(tf.range(n_orb)[:, None, None] < h2_idx[:, 0], tf.int32)[:, 0]
    h2_phase_fac = tf.abs(h2_phase_fac)
    h2_phase_exp = tf.reduce_sum(h2_phase_fac * h2_states, axis=0)
    h2_phase *= (-1) ** tf.abs(h2_phase_exp)

    h2 *= tf.cast(h2_phase, tf.float32)
    h2 = h2[h2_legal]    
    h2_states = tf.boolean_mask(h2_states, h2_legal, axis=1)
    h2_idx = tf.boolean_mask(h2_idx, h2_legal, axis=0)

    h2_states_idx = tf.where(h2_states > 0)
    h2_states_idx = tf.gather(h2_states_idx, tf.argsort(h2_states_idx[:, 1]))[:, 0]
    h2_states_idx = tf.reshape(h2_states_idx, (len(h2), n_e))
    h2_ampls = tf.gather(mo_features, h2_states_idx)

    h2_ampls = tf.linalg.det(tf.einsum("lnfa,lmfa,f,lnm->lnm", h2_ampls, h2_ampls ** 2, mp_one, gram))

    all_states = tf.concat([h1_states, h2_states], axis=1)
    all_ampls_sq = tf.concat([h1_ampls, h2_ampls], axis=0)

    all_ampls_sq_sorted_idx = tf.argsort(all_ampls_sq, direction="DESCENDING")
    all_ampls_sq_sorted = tf.gather(all_ampls_sq, indices=all_ampls_sq_sorted_idx)
    all_states = tf.gather(all_states, all_ampls_sq_sorted_idx, axis=1)
    mask = tf.cast(tf.concat([tf.constant([tf.reduce_max(all_ampls_sq_sorted)]), all_ampls_sq_sorted[:-1]], axis=0) != all_ampls_sq_sorted, dtype=tf.float32)
    all_ampls_sq_sorted *= mask

    top_N_u_states = tf.gather(all_states, tf.math.top_k(all_ampls_sq_sorted, N_u).indices, axis=1)
    top_N_u_states = tf.cast(top_N_u_states, dtype=tf.float32)
    top_N_u_ampls = tf.gather(all_ampls_sq_sorted, tf.math.top_k(all_ampls_sq_sorted, N_u).indices)

    e1 = h1 * h1_ampls
    e2 = h2 * h2_ampls

    denom = tf.gather(mo_features, tf.where(x_vec)[:, 0])
    
    denom = tf.linalg.det(tf.einsum("nfa,mfa,f,nm->nm", denom, denom, mp_one, gram))
    e_loc = (1. * tf.reduce_sum(e1) + 0.5 * tf.reduce_sum(e2)) / denom

    return e_loc, denom, top_N_u_states, top_N_u_ampls


def create_neighbour_states_V_n(mo_features, V_n, h1, h1_idx, h2, h2_idx, n_features, N_u, gram):
    e_loc, denom, top_N_u_states, top_N_u_ampls = tf.map_fn(lambda x: create_neighbour_states(mo_features, x, h1, h1_idx, h2, h2_idx, n_features, N_u, gram), V_n, fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
    top_N_u_ampls_flattened = tf.reshape(top_N_u_ampls, [-1])
    top_N_u_states_shape = tf.shape(top_N_u_states)
    top_N_u_states_flattened = tf.reshape(top_N_u_states, [top_N_u_states_shape[0] * top_N_u_states_shape[1], top_N_u_states_shape[2]])

    all_ampls_sq_sorted_idx = tf.argsort(top_N_u_ampls_flattened, direction="DESCENDING")
    all_ampls_sq_sorted = tf.gather(top_N_u_ampls_flattened, indices=all_ampls_sq_sorted_idx)
    all_states = tf.gather(top_N_u_states_flattened, all_ampls_sq_sorted_idx, axis=1)
    mask = tf.cast(tf.concat([tf.constant([tf.reduce_max(all_ampls_sq_sorted)]), all_ampls_sq_sorted[:-1]], axis=0) != all_ampls_sq_sorted, dtype=tf.float32)
    all_ampls_sq_sorted *= mask

    top_N_u_states = tf.gather(all_states, tf.math.top_k(all_ampls_sq_sorted, N_u).indices, axis=1)
    top_N_u_states = tf.cast(top_N_u_states, dtype=tf.float32)
    #top_N_u_ampls_2 = tf.gather(all_ampls_sq_sorted, tf.math.top_k(all_ampls_sq_sorted, N_u).indices)

    return e_loc, denom, top_N_u_states


def calc_en_batchwise(mo_features, V_n, h1, h1_idx, h2, h2_idx, n_features, N_u, gram):
    inp = (mo_features, V_n, h1, h1_idx, h2, h2_idx, n_features, N_u, gram)
    print(inp)
    res = tf.map_fn(lambda x: create_neighbour_states_V_n(*inp), inp, fn_output_signature=(tf.float32, tf.float32, tf.float32))
    return res