import os
from include.headers import *

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class Visualiser:
    def __init__(self, image_directory='images/'):
        self.IMAGE_DIRECTORY = image_directory
        self.logger = Logger("Visualiser")

    # Function to pre-process data
    def _preprocess_data(self, sample):
        # Normalise the pixel values
        sample = sample / 255.0

        # Flatten the data
        sample = sample.reshape(1, 28, 28, 1)
        
        return sample
    
    # Function to load data sample
    def load_data_sample(self) -> image:
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        
        # Take the first image and pre-process
        x_sample = x_train[0]
        x_sample = self._preprocess_data(x_sample)

        return x_sample
    
    # Function to visualise feature maps
    def visualise_feature_maps(self, model: Model, str_model: str, layer_names: list, image_input) -> None:
        # Get the layers from the model
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        feature_model = Model(inputs=model.inputs, outputs=layer_outputs)
        
        # Initialise feature map
        feature_maps = feature_model.predict(image_input)

        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}{str_model}"
        os.makedirs(result_directory, exist_ok=True)

        for layer_name, feature_map in zip(layer_names, feature_maps):
            self.logger.debug(f"Layer: {layer_name} | Feature Map Shape: {feature_map.shape}")

            if(len(feature_map.shape) != 4):
                self.logger.debug(f"Skipping {layer_name}: Not an image representation")
                continue

            num_filters = feature_map.shape[-1]
            size = feature_map.shape[1]

            plt.figure(figsize=(15, 15))
            plt.suptitle(f"Feature Maps from Layer: {layer_name}")

            for i in range(min(6, num_filters)):
                plt.subplot(1, 6, i + 1)
                plt.imshow(feature_map[0, :, :, i], cmap='viridis')
                plt.axis('off')
            
            plt.savefig(f"{result_directory}/{str_model}_{layer_name}_feature_map.png")

    # Function to visualise filters
    def visualise_grayscale_filters(self, model: Model, str_model: str , layer_name: str):
        conv_layer = model.get_layer(layer_name)
        filters, _ = conv_layer.get_weights()

        num_filters = filters.shape[-1]  # Number of filters
        fig, axes = plt.subplots(1, min(6, num_filters), figsize=(15, 5))

        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}{str_model}"
        os.makedirs(result_directory, exist_ok=True)

        for i in range(min(6, num_filters)):  # Show up to 6 filters
            f = filters[:, :, 0, i]  # Extract one filter (grayscale case)
            f = (f - f.min()) / (f.max() - f.min())  # Normalize to [0,1]
            
            axes[i].imshow(f, cmap='gray')
            axes[i].axis("off")
            axes[i].set_title(f"Filter {i}")

        plt.savefig(f"{result_directory}/{str_model}_{layer_name}_filter.png")
