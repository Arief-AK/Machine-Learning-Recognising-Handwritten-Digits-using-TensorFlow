from include.headers import *
from include.TensorModel import TensorModel
from include.Visualiser import Visualiser

IMAGE_DIRECTORY = 'images/'
NUM_DISPLAYED_SAMPLES = 7
NUM_EPOCHS = 10

# Display some sample images
def display_sample_images(x_train: np.ndarray):
    fig, axes = plt.subplots(1, NUM_DISPLAYED_SAMPLES, figsize=(20, 3))
    for i in range(NUM_DISPLAYED_SAMPLES):
        axes[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f"{IMAGE_DIRECTORY}sample_images.png")

# Determine model
def determine_model(choice: int, logger: Logger) -> tuple:
    str_model: str = ""
    test_acc: float = 0.0
    y_pred: np.ndarray = None
    model: tf.keras.Model
    
    if choice == 1:
        # Get logistic regression model using TensorFlow
        str_model = "Logistic Regression"
        model = model_handler.tensor_create_logistic_regression_model()
    elif choice == 2:
        # Get CNN model using TensorFlow
        str_model = "CNN"
        model = model_handler.tensor_create_cnn_model()
    else:
        pass
    
    # Compile and evaluate model
    model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    y_pred = model_handler.tensor_predict(model, x_test, NUM_DISPLAYED_SAMPLES)
    
    # Log results
    logger.info(f"{str_model} Model Accuracy: {test_acc * 100:.2f}%")

    return (model, str_model), (y_pred, test_acc)

# Display results
def display_results(choice: int, logger: Logger, x_test , y_pred: np.ndarray):
    # Initialise a sub-plot
    fig, axes = plt.subplots(1, NUM_DISPLAYED_SAMPLES, figsize=(10, 3))

    # Determine the model
    for i, ax in enumerate(axes):
        if choice == 1:
            ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
        elif choice == 2:
            ax.imshow(x_test[i], cmap='gray')
        else:
            pass

        ax.set_title(f"Predicted: {np.argmax(y_pred[i])}")
        ax.axis("off")
    
    # Save figure to image directory
    plt.savefig(f"{IMAGE_DIRECTORY}{str_model}_sample_images.png")
    logger.info(f"Saved {str_model} results in {IMAGE_DIRECTORY}{str_model}")

if __name__ == "__main__":
    # Choose the model to use
    choice = int(input("Enter 1 for Logistic Regression, 2 for CNN: "))

    # Initialise handlers
    model_handler = TensorModel(IMAGE_DIRECTORY, NUM_DISPLAYED_SAMPLES)
    logger = Logger("Main")

    # Initialise variables
    str_model: str = ""
    test_acc: float = 0.0
    y_pred: np.ndarray = None
    model: tf.keras.Model
    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = model_handler.load_data(choice)

    # Display some sample images
    display_sample_images(x_train)

    # Determine the model to use
    (model, str_model), (y_pred, test_acc) = determine_model(choice, logger)

    # Display results
    display_results(choice, logger, x_test, y_pred)

    # Display Visualisation
    visualiser = Visualiser()
    sample_image = visualiser.load_data_sample()

    layer_names = ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'flatten', 'dense', 'dense_1']
    visualiser.visualise_feature_maps(model, str_model, layer_names, sample_image)
    
    print("Done!")