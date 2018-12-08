from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self, *, batch_size):
        self._train_data = None
        self._test_data = None

        self._batch_size = batch_size

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    @abstractmethod
    def show(self):
        raise NotImplementedError