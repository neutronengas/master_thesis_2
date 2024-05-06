import tensorflow as tf

def create_neighbour_states(mo_features, x_vec, h1, h1_idx, h2, h2_idx, n_features, N_u):
            x_vec = tf.cast(x_vec, dtype=tf.int32)
            mp_one = tf.cast(tf.math.floormod(tf.range(n_features), 2) * 2 - 1, dtype=tf.float32)
            # mo_features of shape (mo_orbital, feature channels, angular momentum)
            n_orb = len(x_vec)
            n_e = tf.reduce_sum(x_vec)
            # filter to those matrix elements yielding legitimate state vectors
            
            h1_states = tf.repeat(x_vec[:, None], repeats=len(h1_idx), axis=1)

            h1_minus_shift = tf.cast(tf.range(n_orb)[:, None] <= h1_idx[:, 1], dtype=tf.int32) - tf.cast(tf.range(n_orb)[:, None] < h1_idx[:, 1], dtype=tf.int32)
            h1_states -= h1_minus_shift
            h1_legal = tf.reduce_min(h1_states, axis=0) == 0
            h1_plus_shift = tf.cast(tf.range(n_orb)[:, None] <= h1_idx[:, 0], dtype=tf.int32) - tf.cast(tf.range(n_orb)[:, None] < h1_idx[:, 0], dtype=tf.int32)
            h1_states += h1_plus_shift
            h1_legal = tf.logical_and(tf.reduce_max(h1_states, axis=0) == 1, h1_legal)
            
            h1 = h1[h1_legal]
            h1_states = tf.boolean_mask(h1_states, h1_legal, axis=1)
            # h1_ampls of shape (len(h1_idx), n_electrons, feature channels, angular momentum)

            h1_states_idx = tf.where(h1_states > 0)
            h1_states_idx = tf.gather(h1_states_idx, tf.argsort(h1_states_idx[:, 1]))[:, 0]
            h1_states_idx = tf.reshape(h1_states_idx, (n_e, len(h1)))
            h1_ampls = tf.gather(mo_features, h1_states_idx)

            
            h1_ampls = tf.linalg.det(tf.einsum("nlfa,mlfa,f->lnm", h1_ampls, h1_ampls, mp_one))


            h2_states = tf.repeat(x_vec[:, None], repeats=len(h2_idx), axis=1)


            h2_minus_shift = tf.cast(tf.range(n_orb)[:, None] <= h2_idx[:, 3], dtype=tf.int32) - tf.cast(tf.range(n_orb)[:, None] < h2_idx[:, 3], dtype=tf.int32)
            h2_minus_shift += tf.cast(tf.range(n_orb)[:, None] <= h2_idx[:, 2], dtype=tf.int32) - tf.cast(tf.range(n_orb)[:, None] < h2_idx[:, 2], dtype=tf.int32)
            h2_states -= h2_minus_shift
            h2_legal = tf.reduce_min(h2_states, axis=0) == 0

            h2_plus_shift = tf.cast(tf.range(n_orb)[:, None] <= h2_idx[:, 1], dtype=tf.int32) - tf.cast(tf.range(n_orb)[:, None] < h2_idx[:, 1], dtype=tf.int32)
            h2_plus_shift += tf.cast(tf.range(n_orb)[:, None] <= h2_idx[:, 0], dtype=tf.int32) - tf.cast(tf.range(n_orb)[:, None] < h2_idx[:, 0], dtype=tf.int32)
            h2_states += h2_plus_shift
            h2_legal = tf.logical_and(tf.reduce_max(h2_states, axis=0) == 1, h2_legal)
            
            

            # h1_ampls of shape (len(h1_idx), n_electrons, feature channels, angular momentum)
            h2 = h2[h2_legal]
            h2_states = tf.boolean_mask(h2_states, h2_legal, axis=1)
            
            h2_states_idx = tf.where(h2_states > 0)
            h2_states_idx = tf.gather(h2_states_idx, tf.argsort(h2_states_idx[:, 1]))[:, 0]
            h2_states_idx = tf.reshape(h2_states_idx, (n_e, len(h2)))
            h2_ampls = tf.gather(mo_features, h2_states_idx)


            h2_ampls = tf.linalg.det(tf.einsum("nlfa,mlfa,f->lnm", h2_ampls, h2_ampls, mp_one))
            all_states = tf.concat([h1_states, h2_states], axis=1)
            all_ampls = tf.concat([h1_ampls, h2_ampls], axis=0)


            top_N_u_states = tf.gather(all_states, tf.math.top_k(all_ampls, N_u).indices, axis=1)
            top_N_u_states = tf.cast(top_N_u_states, dtype=tf.float32)
            top_N_u_ampls = tf.gather(all_ampls, tf.math.top_k(all_ampls, N_u).indices)


            e1 = h1 * h1_ampls
            e2 = h2 * h2_ampls


            denom = tf.gather(mo_features, tf.where(x_vec)[:, 0])
            denom = tf.linalg.det(tf.einsum("nfa,mfa,f->nm", denom, denom, mp_one))
            e_loc = (tf.reduce_sum(e1) + tf.reduce_sum(e2)) / denom

            return e_loc, denom, top_N_u_states, top_N_u_ampls
        
