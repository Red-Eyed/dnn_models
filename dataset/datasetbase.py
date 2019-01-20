from abc import ABC, abstractmethod


class DatasetBase(ABC):
    def __init__(self):
        self.train_init_op = None
        self.test_init_op = None

        self.x = None
        self.y = None

        self.train_size = None
        self.test_size = None

        self.batch_size = None

    @abstractmethod
    def show(self):
        raise NotImplementedError
