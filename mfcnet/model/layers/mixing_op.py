from tensorflow.keras import layers
import tensorflow as tf

from ..layers.hermite_layer import HermiteLayer


class MixingLayer(layers.Layer):
    def __init__(self, n_features, lmax, n_mlp, k, activation=None, name='pairmix', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.activation = activation
        self.k = k
        self.lmax = lmax + 1
        self.n_mlp = n_mlp
        self.hermite_layer = HermiteLayer(n_features=n_features)


    def build(self, shape):
        self.wx = []
        self.bx = []
        self.wy = []
        self.by = []
        self.wx.append(self.add_weight(name=f"wx_0", shape=(self.k, self.n_features, self.n_features, self.lmax), initializer=self.initializer))
        self.wy.append(self.add_weight(name=f"wy_0", shape=(self.k, self.n_features, self.n_features, self.lmax), initializer=self.initializer))
        for n in range(self.n_mlp):
            self.wx.append(self.add_weight(name=f"wx_{n+1}", shape=(self.k, self.k), initializer=self.initializer))
            self.bx.append(self.add_weight(name=f"bx_{n}", shape=(self.k, self.k), initializer=self.initializer))
            self.wy.append(self.add_weight(name=f"wy_{n+1}", shape=(self.k, self.k), initializer=self.initializer))
            self.by.append(self.add_weight(name=f"by_{n}", shape=(self.k, self.k), initializer=self.initializer))
        self.wx.append(self.add_weight(name="wx_f", shape=(self.lmax, self.n_features, self.k), initializer=self.initializer))
        self.wy.append(self.add_weight(name="wy_f", shape=(self.lmax, self.n_features, self.k), initializer=self.initializer))

    def call(self, inputs):
        # x, y shape: (batch, mo, features, ang mom)
        x, y = inputs
        mix = x[:, :, None, :, :] * y[:, :, :, None, :]
        mix = tf.transpose(mix(4, 1, 2, 3, 0))
        mix = tf.math.unsorted_segment_sum(mix, tf.repeat(tf.range(self.lmax), 2 * tf.range(self.lmax) + 1))
        mix = tf.transpose(mix(4, 1, 2, 3, 0))
        mix_x = tf.einsum("bmfgl,kfgl->bmk", mix, self.wx[0])
        mix_y = tf.einsum("bmfgl,kfgl->bmk", mix, self.wy[0])
        for i in range(1, self.n_mlp):
            mix_x = self.activation(tf.einsum("bmk,kl->bml", mix_x, self.wx[i]) + self.bx[i])
            mix_y = self.activation(tf.einsum("bmk,kl->bml", mix_y, self.wy[i]) + self.by[i])
        mix_x = self.activation(tf.einsum("bmk,lfk->bmfl", mix_x, self.wx[i]))
        mix_y = self.activation(tf.einsum("bmk,lfk->bmfl", mix_y, self.wy[i]))
        res = tf.repeat(mix_x, repeats=(2*tf.range(self.lmax) + 1), axis=-1) * x 
        res += tf.repeat(mix_y, repeats=(2*tf.range(self.lmax) + 1), axis=-1) * y
        return res