from tensorflow.keras import layers
import tensorflow as tf

class TensorProductExpansionLayer(layers.Layer):
    def __init__(self, cgc, name='tensprodexp', **kwargs):
        super().__init__(name=name, **kwargs)

        self.cgc = cgc

    def set_params(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def call(self, inputs):
        [x] = inputs
        xl3_embedding = x[self.l3]

        if (self.l1, self.l2, self.l3) in self.cgc.keys():
            l1l2l3_cgc = self.cgc[(self.l1, self.l2, self.l3)]
        else:
            l1l2l3_cgc = tf.zeros(shape=(2*self.l1+1, 2*self.l2+1, 2*self.l3+1))
        res = tf.einsum("mnh,ifh->imnf", l1l2l3_cgc, xl3_embedding)
        return res