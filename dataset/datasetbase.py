from abc import ABC, abstractmethod


class DatasetBase(ABC):
    def __init__(self, batch_size, prefetch_size, shuffle_size):
        self.batch_size = batch_size
        self.prefetch_size = int(prefetch_size)
        self.shuffle_size = int(shuffle_size)

        self.train_init_op = None
        self.test_init_op = None

        self.x = None
        self.y = None

        self.train_size = None
        self.test_size = None


    @abstractmethod
    def show(self):
        raise NotImplementedError
