import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class TensorModel:
    def __init__(self, image_directory='images/', num_displayed_samples=7):
        self.IMAGE_DIRECTORY = image_directory
        self.NUM_DISPLAYED_SAMPLES = num_displayed_samples

        # Ensure GPU is available
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    def _preprocess_data_linear_reg(self, x_train: np.ndarray, x_test: np.ndarray) -> tuple:
        # Normalise pixel values
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Flatten the data from a 2D (28x28) to 1D (728) array
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        return x_train_flat, x_test_flat

    def _preprocess_data_cnn(self, x_train: np.ndarray, x_test: np.ndarray) -> tuple:
        # Normalise pixel values
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Flatten the data to single channel
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        return x_train, x_test

    def load_data(self, choice: int) -> tuple:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        if choice == 1:
            x_train, x_test = self._preprocess_data_linear_reg(x_train, x_test)
        elif choice == 2:
            x_train, x_test = self._preprocess_data_cnn(x_train, x_test)
        return (x_train, y_train), (x_test, y_test)

    def tensor_create_logistic_regression_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def tensor_create_cnn_model(self) -> tf.keras.Model:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def tensor_predict(self, model: tf.keras.Model, x_test: np.ndarray, num_samples=5) -> np.ndarray:
        y_pred = model.predict(x_test[:num_samples])
        return y_pred