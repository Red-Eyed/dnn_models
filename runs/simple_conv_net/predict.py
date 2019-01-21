#!/usr/bin/env python3

from config import Config
from dataset.keras_dataset import MNIST
from models.simple_conv_net import SimpleConvNet
from utils.tf_utils import predict

if __name__ == '__main__':
    dataset = MNIST(batch_size=1, prefetch_size=10, shuffle_size=10)
    model = SimpleConvNet(dataset.x, dataset.y)

    predict(model, dataset, restore_path=Config.MODELS_DIR / str(model) / str(model))
