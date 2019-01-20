#!/usr/bin/env python3

from dataset.mnist import MNIST
from models.simple_conv_net import ConvNet

from tf_utils import fit

if __name__ == '__main__':
    dataset = MNIST(batch_size=100)
    model = ConvNet(dataset.x, dataset.y)

    fit(model, dataset)
