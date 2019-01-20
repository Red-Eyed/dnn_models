from abc import abstractmethod

from tensorflow.python.data import Iterator, Dataset
import tensorflow as tf
import numpy as np
from skimage import io

from dataset.datasetbase import DatasetBase


class _KerasDataset(DatasetBase):
    def __init__(self, *, batch_size, num_classes):
        super().__init__()
        self._num_classes = num_classes
        self.batch_size = batch_size
        (x_train, y_train), (x_test, y_test) = self._load_data()

        x_train = np.float32(x_train)
        x_test = np.float32(x_test)

        y_train = np.float32(self._to_one_hot(y_train))
        y_test = np.float32(self._to_one_hot(y_test))

        self.train_size = y_train.shape[0]
        self.test_size = y_test.shape[0]

        output_types = (x_train.dtype, y_train.dtype)
        output_shape = (x_train.shape[1:], y_train.shape[1:])

        train_dataset = Dataset.from_generator(lambda: self._gen_data(x_train, y_train), output_types, output_shape). \
            batch(self.batch_size, drop_remainder=True)

        test_dataset = Dataset.from_generator(lambda: self._gen_data(x_test, y_test), output_types, output_shape). \
            batch(self.batch_size, drop_remainder=True)

        iter_ = Iterator.from_structure(train_dataset.output_types,
                                        train_dataset.output_shapes)

        self.x, self.y = iter_.get_next()

        self.train_init_op = iter_.make_initializer(train_dataset)
        self.test_init_op = iter_.make_initializer(test_dataset)

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def _to_one_hot(self, inputs):
        one_hot = tf.keras.utils.to_categorical(inputs, self._num_classes)
        # one_hot = np.zeros((inputs.size, self._num_classes))
        # one_hot[np.arange(inputs.size), inputs] = 1

        return one_hot

    def _gen_data(self, x, y):
        assert len(x) == len(y)

        idx = np.arange(0, x.shape[0], 1)
        np.random.shuffle(idx)

        np.take(x, indices=idx, axis=0, out=x)
        np.take(y, indices=idx, axis=0, out=y)

        for x, y in zip(x, y):
            yield x, y

    def show(self):
        np.random.seed(0)
        tf.random.set_random_seed(0)

        train = True
        test = False
        switch = True

        with tf.Session() as sess:
            while True:
                # Switch to test or train dataset
                if train and switch:
                    switch = False
                    sess.run(self.train_init_op)
                elif test and switch:
                    switch = False
                    sess.run(self.test_init_op)

                try:
                    # Get train or test data
                    x, y = sess.run([self.x, self.y])
                except tf.errors.OutOfRangeError:
                    train = not train
                    test = not test
                    switch = True
                    continue

                for x, y in zip(x, y):
                    print(y)
                    if x.shape[-1] == 1:
                        io.imshow(x[:, :, 0], cmap="gray")
                    elif x.shape[-1] == 3:
                        io.imshow(x)
                    else:
                        raise ValueError

                    io.show()


class MNIST(_KerasDataset):
    def __init__(self, *, batch_size):
        super().__init__(batch_size=batch_size, num_classes=10)

    @staticmethod
    def _load():
        from tensorflow.python.keras.datasets import mnist as dataloader
        return dataloader.load_data()

    def _load_data(self):
        out = [list(i) for i in self._load()]
        for i in range(len(out)):
            for j in range(len(out[i])):
                out[i][j] = out[i][j].reshape((*out[i][j].shape, 1))
                out[i][j].flags['WRITEABLE'] = True

        return out


class FashionMNIST(MNIST):
    @staticmethod
    def _load():
        from tensorflow.python.keras.datasets import fashion_mnist as dataloader
        return dataloader.load_data()


class CIFAR10(_KerasDataset):
    def __init__(self, *, batch_size):
        super().__init__(batch_size=batch_size, num_classes=10)

    def _load_data(self):
        from tensorflow.python.keras.datasets import cifar10 as dataloader
        return dataloader.load_data()


class CIFAR100(_KerasDataset):
    def __init__(self, *, batch_size):
        super().__init__(batch_size=batch_size, num_classes=100)

    def _load_data(self):
        from tensorflow.python.keras.datasets import cifar100 as dataloader
        return dataloader.load_data()


if __name__ == '__main__':
    dataset = CIFAR100(batch_size=10)
    dataset.show()
