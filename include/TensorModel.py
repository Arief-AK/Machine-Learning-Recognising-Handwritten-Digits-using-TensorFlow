from include.headers import *

class TensorModel:
    def __init__(self, image_directory='images/', num_displayed_samples=7):
        self.IMAGE_DIRECTORY = image_directory
        self.NUM_DISPLAYED_SAMPLES = num_displayed_samples
        self.logger = Logger("TensorModel")

        # Ensure GPU is available
        self.logger.info(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")

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
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),     # Creates 32 kernels of size 3x3, relu activation used to remove negative values, detects basic patterns
            MaxPooling2D((2, 2)),                                               # Selects maximum value in 2x2 region, reduces size of feature
            Conv2D(64, (3, 3), activation='relu'),                              # Creates 64 kernels of size 3x3, detects complex patterns (corners, shapes)
            MaxPooling2D((2, 2)),                                               # Selects maximum values in 2x2 regions
            Flatten(),                                                          # Flattens 2D feature maps into 1D vector, enables utilising Dense layers
            Dense(128, activation='relu'),                                      # Learn abstract representations, 128 neurons
            Dense(10, activation='softmax')                                     # Dataset has 10 digit classes (0-9), uses softmax to convert output to probabilities
        ])

        # Model Compilation
        # Optimizer: Updates model weights to minimise loss
        # adam: Adaptive Moment Estimation
        #       Adaptive learning rate:     Adjust each parameter, preventing slow learning
        #       Fast convergence:           Reaches good accuracy quickly
        #
        # Loss: Measures how wrong the predictions are, tells optmizer how to adjust weights
        # sparse_categorial_crossentropy
        #       Multi-class classification: MNIST has 10 classes (0-9)
        #
        # Metrics:
        # accuracy
        #       Measures how many predictions were correct
        #       Provides percentage of correct classification
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def tensor_predict(self, model: tf.keras.Model, x_test: np.ndarray, num_samples=5) -> np.ndarray:
        y_pred = model.predict(x_test[:num_samples])
        return y_pred