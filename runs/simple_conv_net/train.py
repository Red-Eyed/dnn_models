#!/usr/bin/env python3

from dataset.keras_dataset import MNIST
from models.simple_conv_net import SimpleConvNet

from utils.tf_utils import fit

if __name__ == '__main__':
    dataset = MNIST(batch_size=100)
    model = SimpleConvNet(dataset.x, dataset.y, dropout_rate=0.5, learning_rate=0.001)

    fit(model, dataset, save_interval_minute=2, epochs=10)
