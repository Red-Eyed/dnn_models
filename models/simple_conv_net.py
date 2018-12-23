from models.base import Model

import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten


class ConvNet(Model):
    def __init__(self, x, y, num_classes=10):
        super().__init__(x, y)
        self._num_classes = num_classes

    def _logits_internal(self):
        x = self._x

        x = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu",
                   kernel_initializer=tf.initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu",
                   kernel_initializer=tf.initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu",
                   kernel_initializer=tf.initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(self._num_classes)(x)

        return x

    def _loss_internal(self):
        return tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self.logits())

    def _optimize(self, *, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss())

    def _predict_internal(self):
        probs = tf.nn.softmax(self._logits_internal())
        return probs
