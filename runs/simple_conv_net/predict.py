#!/usr/bin/env python3
from pathlib import Path

from dataset.keras_dataset import MNIST
from models.simple_conv_net import SimpleConvNet

from utils.tf_utils import predict

if __name__ == '__main__':
    dataset = MNIST(batch_size=1)
    model = SimpleConvNet(dataset.x, dataset.y)

    predict(model, dataset, load_path=Path("./") / str(model) / str(model))
