from models.base import Model

import tensorflow as tf


class Classifier(Model):
    def __init__(self, x, y, num_classes=10, dropout_rate=0., learning_rate=0.001):
        super().__init__(x, y, dropout_rate, learning_rate)
        self._num_classes = num_classes

    def _logits_impl(self):
        raise NotImplementedError

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
