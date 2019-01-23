from models.base import Model

import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, ReLU


class SimpleConvNet(Model):
    def __init__(self, x, y, num_classes=10, dropout_rate=0., learning_rate=0.001):
        super().__init__(x, y, dropout_rate, learning_rate)
        self._num_classes = num_classes

    def _logits_impl(self):
        x = self._x

        x = Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(100)(x)
        x = ReLU()(x)

        x = Dense(self._num_classes)(x)

        return x

    def _loss_impl(self):
        return tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self.logits())

    def _optimize_impl(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss())

    def _predict_impl(self):
        preds = tf.nn.softmax(self.logits())
        return preds

    def _accuracy_impl(self):
        preds = tf.argmax(self.predict(), axis=-1)
        labels = tf.argmax(self._y, axis=-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return acc

