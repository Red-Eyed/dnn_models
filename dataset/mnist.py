from tensorflow.python.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from skimage import io

from dataset.base import Base


class MNIST(Base):
    def __init__(self, *, batch_size):
        super().__init__(batch_size=batch_size)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        output_types = (x_train.dtype, y_train.dtype)
        output_shape = (x_train.shape[1:], y_train.shape[1:])

        self._train_data = tf.data.Dataset.from_generator(lambda: self._gen_data(x_train, y_train), output_types,
                                                          output_shape). \
            batch(self._batch_size).make_initializable_iterator()

        self._test_data = tf.data.Dataset.from_generator(lambda: self._gen_data(x_test, y_test), output_types,
                                                         output_shape). \
            batch(self._batch_size).make_initializable_iterator()

    def _gen_data(self, x, y):
        assert len(x) == len(y)

        idx = np.arange(0, x.shape[0], 1)
        np.random.shuffle(idx)

        np.take(x, indices=idx, axis=0, out=x)
        np.take(y, indices=idx, axis=0, out=y)

        for x, y in zip(x, y):
            yield x, y

    def show(self):
        while True:
            with tf.Session() as sess:
                sess.run(self.train_data.initializer)
                while True:
                    x_batch, y_batch = sess.run(self.train_data.get_next())
                    for x, y in zip(x_batch, y_batch):
                        print(y)
                        io.imshow(x)
                        io.show()


if __name__ == '__main__':
    dataset = MNIST(batch_size=10)
    dataset.show()
