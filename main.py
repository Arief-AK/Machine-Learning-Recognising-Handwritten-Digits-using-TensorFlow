import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

IMAGE_DIRECTORY = 'images/'

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

if __name__ == "__main__":
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Display some sample images
    display_sample_images(x_train)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial')
    model.fit(x_train, y_train)

    # Predict on the test data
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")