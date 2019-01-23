from pathlib import Path
from skimage import io
import tensorflow as tf

from datetime import datetime

from tqdm import tqdm

from dataset.datasetbase import DatasetBase
from models.base import Model
import numpy as np
import logging


def get_num_of_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1

        for dim in shape:
            variable_parameters *= dim.value

        total_parameters += variable_parameters

    return total_parameters


def fit(model: Model, dataset: DatasetBase, save_path: Path, save_interval_minute: int = 15, epochs: int = 1):
    save_path = save_path.expanduser().absolute()

    model.build_graph()

    train = True
    test = not train
    switch = True
    sum_loss = 0
    sum_acc = 0
    saver = tf.train.Saver(save_relative_paths=True)

    epoch = -1

    now = datetime.now().minute

    logging.info(f"Number of trainable parameters: {get_num_of_parameters()}")

    with tf.Session() as sess, tqdm() as progress:
        sess.run(tf.global_variables_initializer())
        sum_writer = tf.summary.FileWriter(save_path / str(model) / "logdir", sess.graph)
        while epoch <= epochs:
            # Switch to test or train dataset
            if switch:
                switch = False

                if train:
                    sess.run(dataset.train_init_op)
                    progress.total = dataset.train_size
                    epoch += 1
                elif test:
                    sess.run(dataset.test_init_op)
                    progress.total = dataset.test_size

            try:
                phase = 'train' if train else 'test'
                loss = 0
                acc = 0
                if phase == 'train':
                    loss, acc, _, = sess.run([model.loss(), model.accuracy(), model.optimize()],
                                             feed_dict={tf.keras.backend.learning_phase(): 1})
                elif phase == 'test':
                    loss, acc = sess.run([model.loss(), model.accuracy()],
                                         feed_dict={tf.keras.backend.learning_phase(): 0})

                sum_loss += loss
                sum_acc += acc

                batches = (progress.n / dataset.batch_size + 1)
                desc = f"Epoch: {epoch:<5}| Phase: {phase :<10}| " \
                    f"loss: {sum_loss / batches :<25}| " \
                    f"acc:  {sum_acc / batches :<25}| "

                progress.set_description(desc=desc)
                progress.update(dataset.batch_size)

            except tf.errors.OutOfRangeError:
                progress.write("")

                train = not train
                test = not test
                switch = True
                progress.n = 0
                sum_loss = 0
                sum_acc = 0

                if now - datetime.now().minute >= save_interval_minute:
                    now = datetime.now().minute
                    saver.save(sess, str(save_path / str(model) / str(model)))
                    sum_writer.flush()
                    sum_writer.close()
                    sum_writer.reopen()

                continue

        sum_writer.flush()
        sum_writer.close()
        saver.save(sess, str(save_path / str(model) / str(model)))


def predict(model: Model, dataset: DatasetBase, restore_path: Path):
    restore_path = restore_path.expanduser().absolute()

    model.build_graph()
    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(restore_path))

        while True:
            sess.run(dataset.test_init_op)

            try:
                pred, x, y = sess.run([model.predict(), dataset.x, dataset.y],
                                      feed_dict={tf.keras.backend.learning_phase(): 0})
                pred = np.argmax(pred)
                print(pred)
                io.imshow(x[0, :, :, 0], cmap="gray")
                io.show()

            except tf.errors.OutOfRangeError:
                continue
