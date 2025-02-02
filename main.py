import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

IMAGE_DIRECTORY = 'images/'
USE_TENSORFLOW = True

# Preprocess the data
def preprocess_data(x_train: np.ndarray, x_test: np.ndarray) -> tuple:
    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert the 2D images to 1D of size 784
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Log the shapes of the data
    print(f"Shape of x_train_flat: {x_train_flat.shape}")
    print(f"Shape of x_test_flat: {x_test_flat.shape}")

    return x_train_flat, x_test_flat

# Load the dataset
def load_data() -> tuple:
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess the data
    x_train_flat, x_test_flat = preprocess_data(x_train, x_test)

    return (x_train_flat, y_train), (x_test_flat, y_test)

# Display some sample images
def display_sample_images(x_train: np.ndarray):
    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    for i in range(10):
        axes[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f"{IMAGE_DIRECTORY}sample_images.png")

# Create logistic regression model
def create_logistic_regression_model(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial')
    model.fit(x_train, y_train)

    return model

# Create logistic regression model using TensorFlow
def tensor_create_logistic_regression_model(x_train: np.ndarray, y_train: np.ndarray) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Display some sample images
    display_sample_images(x_train)

    # Ensure GPU is available
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    if(USE_TENSORFLOW == False):
        # Get logistic regression model using scikit-learn
        logistic_regression_model = create_logistic_regression_model(x_train, y_train)
        y_pred = logistic_regression_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

    # Get logistic regression model using TensorFlow
    tensor_model = tensor_create_logistic_regression_model(x_train, y_train)
    tensor_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    test_loss, test_acc = tensor_model.evaluate(x_test, y_test)
    print(f"TensorFlow Model Accuracy: {test_acc * 100:.2f}%")