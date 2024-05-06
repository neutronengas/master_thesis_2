from tensorflow.keras import layers
import tensorflow as tf

class LinearLayer(layers.Layer):
    def __init__(self, n_features, name='linear', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_features = n_features


    def build(self, shape):
        initializer = tf.keras.initializers.GlorotNormal()
        self.W = self.add_weight(name="W", shape=(self.n_features, self.n_features), dtype=tf.float32, initializer=initializer, trainable=True)
        self.b = self.add_weight(name="b", shape=(self.n_features,), initializer=initializer)

        
    def call(self, inputs):
        # shape (batch, molecular orbital, feature, angular momentum)
        mo_features = inputs
        mo_features = tf.einsum("abcd,ic->abid", mo_features, self.W)
        mo_features[:, :, :, 0] += self.b
        return mo_features
