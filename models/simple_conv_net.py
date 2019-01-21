from models.base import Model

import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Dropout


class SimpleConvNet(Model):
    def __init__(self, x, y, num_classes=10, dropout_rate=0., learning_rate=0.001):
        super().__init__(x, y, dropout_rate, learning_rate)
        self._num_classes = num_classes

    def _logits_internal(self):
        x = self._x

        x = BatchNormalization()(x)
        x = Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding="valid", activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(self.dropout_rate)(x)

        x = BatchNormalization()(x)
        x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="valid", activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(self.dropout_rate)(x)

        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)

        x = BatchNormalization()(x)
        x = Dense(self._num_classes)(x)

        return x

    def _loss_internal(self):
        return tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self.logits())

    def _optimize(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss())

    def _predict_internal(self):
        probs = tf.nn.softmax(self.logits())
        return probs
