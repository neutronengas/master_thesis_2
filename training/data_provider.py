from collections import OrderedDict
import numpy as np
import tensorflow as tf

class DataProvider:
    def __init__(self, N_u, data_container, ntrain, nvalid, batch_size=1, seed=None, randomized=False):

        self.data_container = data_container
        self._ndata = len(data_container)
        self.nsamples = {"train": ntrain, "val": nvalid, "test": len(data_container) - ntrain - nvalid}
        self.batch_size = batch_size
        self.N_u = N_u
        # Random state parameter such that random operations are reproducible
        self._random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(self.data_container))
        if randomized:
            all_idx = self._random_state.permutation(all_idx)

        self.idx = {"train": all_idx[0:ntrain],
                           "val": all_idx[ntrain:ntrain+nvalid],
                           "test": all_idx[ntrain+nvalid:]}
        
        self.idx_in_epoch = {"train": 0, "val": 0, "test": 0}

        self.dtypes_input = OrderedDict()
        self.dtypes_input["idx"] = tf.int32
        self.dtypes_input["Z"] = tf.int32
        self.dtypes_input["R"] = tf.float32
        self.dtypes_input["C"] = tf.float32
        self.dtypes_input["S"] = tf.float32
        self.dtypes_input["mels"] = tf.float32
        self.dtypes_input["V_0_dec_list"] = tf.int32
        self.dtypes_input["nb_dets"] = tf.int32
        self.dtypes_input["mo_neighbours_i"] = tf.int32
        self.dtypes_input["mo_neighbours_j"] = tf.int32
        self.dtypes_input["coupl"] = tf.float32
        self.dtypes_input["gram"] = tf.float32
        self.dtype_target = tf.float32


        self.shapes_input = {}
        self.shapes_input["idx"] = [None]
        self.shapes_input["Z"] = [None, 2]
        self.shapes_input["R"] = [None, 10, 2, 3]
        self.shapes_input["C"] = [None, 10, 2, 5]
        self.shapes_input["S"] = [None, 2, 3, 2, 3]
        self.shapes_input["mels"] = [None, 14400, 610]
        self.shapes_input["V_0_dec_list"] = [None, 200, 2]
        self.shapes_input["nb_dets"] = [None, 40000]
        self.shapes_input["mo_neighbours_i"] = [None, None]
        self.shapes_input["mo_neighbours_j"] = [None, None]
        self.shapes_input["coupl"] = [None]
        self.shapes_input["gram"] = [None, 10, 10]
        self.shape_target = [None, 2]

    def shuffle_train(self):
        # Shuffle the training data
        self.idx["train"] = self._random_state.permutation(self.idx["train"])

    def get_batch_idx(self, split):
        # Return the indices for a batch of samples from the specified set
        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        if start == 0 and split == "train":
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.idx[split][start:end]
    
    def idx_to_data(self, idx, return_flattened=False):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_container[idx]

        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            #inputs_targets.append(tf.constant(batch["target"]))
            return inputs_targets
        else:
            inputs = {}
            for key, dtype in self.dtypes_input.items():
                inputs[key] = tf.constant(batch[key], dtype=dtype)
            targets = tf.constant(batch['target'], dtype=tf.float32)
            return inputs

    def get_dataset(self, split):
        """Get a generator-based tf.dataset"""
        def generator():
            while True:
                idx = self.get_batch_idx(split)
                yield self.idx_to_data(idx)
        return tf.data.Dataset.from_generator(
            generator,
            output_types=(dict(self.dtypes_input), self.dtype_target),
            output_shapes=(self.shapes_input, self.shape_target)
        )
    
    def get_idx_dataset(self, split):
        """Get a generator-based tf.dataset returning just the indices"""
        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)
        return tf.data.Dataset.from_generator(
            generator,
            output_types=tf.int32,
            output_shapes=[None]
        )
        
    def idx_to_data_tf(self, idx):
        """Convert a batch of indices to a batch of data from TensorFlow"""
        dtypes_flattened = list(self.dtypes_input.values())
        dtypes_flattened.append(self.dtype_target)

        inputs_targets = tf.py_function(lambda idx: self.idx_to_data(idx.numpy(), return_flattened=True),
                                        inp=[idx], Tout=dtypes_flattened)

        inputs = {}

        for i, key in enumerate(self.dtypes_input.keys()):
            inputs[key] = inputs_targets[i]
            inputs[key].set_shape(self.shapes_input[key])
        targets = inputs_targets[-1]
        targets.set_shape(self.shape_target)
        return inputs