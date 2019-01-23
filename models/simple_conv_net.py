from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, ReLU

from models.classifier import Classifier


class SimpleConvNet(Classifier):
    def _logits_impl(self):
        x = self._x

        x = Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(100)(x)
        x = ReLU()(x)

        x = Dense(self._num_classes)(x)

        return x
