import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

# Preprocess data for CNN
def preprocess_data_cnn(x_train: np.ndarray, x_test: np.ndarray) -> tuple:
    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert the 2D images to 3D of size 28x28x1
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Log the shapes of the data
    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of x_test: {x_test.shape}")

    return x_train, x_test

# Load the dataset
def load_data(choice: int) -> tuple:
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess the data
    if(choice == 1):
        x_train_flat, x_test_flat = preprocess_data(x_train, x_test)
    elif(choice == 2):
        x_train_flat, x_test_flat = preprocess_data_cnn(x_train, x_test)
    else:
        pass

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
def tensor_create_logistic_regression_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

# Create CNN model using TensorFlow
def create_cnn_model() -> tf.keras.Model:
    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # 32 filters, 3x3 kernel
        MaxPooling2D((2, 2)),                                           # Reduce the size of the image by half
        Conv2D(64, (3, 3), activation='relu'),                          # 64 filters, 3x3 kernel
        MaxPooling2D((2, 2)),                                           # Reduce the size of the image by half
        Flatten(),                                                      # Flatten the 2D array to 1D
        Dense(128, activation='relu'),                                  # Fully connected layer with 128 units
        Dense(10, activation='softmax')                                 # Output layer with 10 units
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

if __name__ == "__main__":
    # Choose the model to use
    choice = int(input("Enter 1 for Logistic Regression, 2 for CNN: "))
    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_data(choice)

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

    # Determine the model to use
    str_model = ""
    test_acc = 0.0
    if choice == 1:
        # Get logistic regression model using TensorFlow
        str_model = "Logistic Regression"
        tensor_model = tensor_create_logistic_regression_model()
        tensor_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
        test_loss, test_acc = tensor_model.evaluate(x_test, y_test)
    elif choice == 2:
        # Get CNN model using TensorFlow
        str_model = "CNN"
        cnn_model = create_cnn_model()
        cnn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
        test_loss, test_acc = cnn_model.evaluate(x_test, y_test)
    else:
        pass

    print(f"{str_model} Model Accuracy: {test_acc * 100:.2f}%")
    print("Done!")