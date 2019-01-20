from pathlib import Path

import tensorflow as tf

from datetime import datetime

from tqdm import tqdm

from dataset.datasetbase import DatasetBase
from models.base import Model


def fit(model: Model, dataset: DatasetBase, save_interval_minute: int = 15, save_path: Path = Path("./")):
    save_path = save_path.expanduser().absolute()

    model.build_graph()

    train = True
    test = not train
    switch = True
    sum_loss = 0
    saver = tf.train.Saver(save_relative_paths=True)

    epoch = -1

    now = datetime.now().minute

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
                desc = f"Epoch: {epoch:<5}| Phase: {phase :<10}| loss: {sum_loss / (progress.n + dataset.batch_size) :<25}"

                progress.set_description(desc=desc)
                progress.update(dataset.batch_size)

            except tf.errors.OutOfRangeError:
                progress.write("")

                train = not train
                test = not test
                switch = True
                progress.n = 0
                sum_loss = 0

                if now - datetime.now().minute >= save_interval_minute:
                    now = datetime.now().minute
                    saver.save(sess, str(save_path / str(model) / str(model)))

                continue
