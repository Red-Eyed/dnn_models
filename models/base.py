from abc import ABC, abstractmethod
from itertools import dropwhile
from typing import Callable
import tensorflow as tf


class Model(ABC):
    def __init__(self, x: tf.Tensor, y: tf.Tensor, dropout_rate=0., learning_rate=0.001):
        self._x = x
        self._y = y

        self.dropout_rate = tf.placeholder_with_default(float(dropout_rate), shape=(), name="drop_out_rate")
        self.learning_rate = learning_rate
        self._logits_op = None
        self._loss_op = None
        self._optimise_op = None
        self._predict_op = None

        self._func_loss = self.loss
        self._func_optimize = self.optimize

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def _logits_internal(self):
        raise NotImplementedError

    @abstractmethod
    def _loss_internal(self):
        raise NotImplementedError

    @abstractmethod
    def _optimize(self):
        raise NotImplementedError

    @abstractmethod
    def _predict_internal(self):
        raise NotImplementedError

    def set_loss(self, func: Callable):
        self._func_loss = func

    def set_optimizer(self, func: Callable):
        self._func_optimize = func

    def logits(self):
        if self._logits_op is None:
            self._logits_op = self._logits_internal()

        return self._logits_op

    def loss(self):
        if self._loss_op is None:
            self._loss_op = self._loss_internal()

        return self._loss_op

    def optimize(self):
        if self._optimise_op is None:
            self._optimise_op = self._optimize()

        return self._optimise_op

    def predict(self):
        if self._predict_op is None:
            self._predict_op = self._predict_internal()

        return self._predict_op

    def build_graph(self):
        self.logits()
        self.loss()
        self.optimize()
        self.predict()

