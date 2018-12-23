#!/usr/bin/env python3

from dataset.mnist import MNIST
from models.simple_conv_net import ConvNet

from tqdm import tqdm
import tensorflow as tf

if __name__ == '__main__':
    dataset = MNIST(batch_size=100)
    model = ConvNet(dataset.x, dataset.y)
    model.build_graph()

    train = True
    test = not train
    switch = True
    sum_loss = 0

    epoch = -1

    with tf.Session() as sess, tqdm() as progress:
        sess.run(tf.global_variables_initializer())
        while True:
            # Switch to test or train dataset
            if train and switch:
                switch = False
                sess.run(dataset.train_init_op)
                progress.total = dataset.train_size
                epoch += 1
            elif test and switch:
                switch = False
                sess.run(dataset.test_init_op)
                progress.total = dataset.test_size

            try:
                loss, _ = sess.run([model.loss(), model.optimize()])
                sum_loss += loss

                phase = 'train' if train else 'test'
                desc = f"#{epoch} Phase: {phase}: loss = {sum_loss / (progress.n + dataset.batch_size) :<25}"

                progress.set_description(desc=desc)
                progress.update(dataset.batch_size)

            except tf.errors.OutOfRangeError:
                train = not train
                test = not test
                switch = True
                progress.n = 0
                sum_loss = 0
                continue
