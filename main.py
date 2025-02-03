import numpy as np
import matplotlib.pyplot as plt

from include.TensorModel import TensorModel

IMAGE_DIRECTORY = 'images/'
NUM_DISPLAYED_SAMPLES = 7

# Display some sample images
def display_sample_images(x_train: np.ndarray):
    fig, axes = plt.subplots(1, NUM_DISPLAYED_SAMPLES, figsize=(20, 3))
    for i in range(NUM_DISPLAYED_SAMPLES):
        axes[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f"{IMAGE_DIRECTORY}sample_images.png")

if __name__ == "__main__":
    # Choose the model to use
    choice = int(input("Enter 1 for Logistic Regression, 2 for CNN: "))

    # Initialise model handler
    model_handler = TensorModel(IMAGE_DIRECTORY, NUM_DISPLAYED_SAMPLES)
    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = model_handler.load_data(choice)

    # Display some sample images
    display_sample_images(x_train)

    # Determine the model to use
    str_model: str = ""
    test_acc: float = 0.0
    y_pred: np.ndarray = None
    if choice == 1:
        # Get logistic regression model using TensorFlow
        str_model = "Logistic Regression"
        logistic_reg_model = model_handler.tensor_create_logistic_regression_model()
        logistic_reg_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
        test_loss, test_acc = logistic_reg_model.evaluate(x_test, y_test)
        y_pred = model_handler.tensor_predict(logistic_reg_model, x_test, NUM_DISPLAYED_SAMPLES)
    elif choice == 2:
        # Get CNN model using TensorFlow
        str_model = "CNN"
        cnn_model = model_handler.tensor_create_cnn_model()
        cnn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
        test_loss, test_acc = cnn_model.evaluate(x_test, y_test)
        y_pred = model_handler.tensor_predict(cnn_model, x_test, NUM_DISPLAYED_SAMPLES)
    else:
        pass
    print(f"{str_model} Model Accuracy: {test_acc * 100:.2f}%")

    # Display results
    fig, axes = plt.subplots(1, NUM_DISPLAYED_SAMPLES, figsize=(10, 3))
    for i, ax in enumerate(axes):
        # Determine the model
        if choice == 1:
            ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
        elif choice == 2:
            ax.imshow(x_test[i], cmap='gray')
        else:
            pass

        ax.set_title(f"Predicted: {np.argmax(y_pred[i])}")
        ax.axis("off")
    plt.savefig(f"{IMAGE_DIRECTORY}{str_model}_sample_images.png")
    
    print("Done!")