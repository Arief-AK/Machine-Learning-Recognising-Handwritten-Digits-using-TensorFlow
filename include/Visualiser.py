from headers import *

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class Visualiser:
    def __init__(self, image_directory='images/'):
        self.IMAGE_DIRECTORY = image_directory

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
        feature_model = Model(inputs=model.input, outputs=layer_outputs)

        feature_maps = feature_model.predict(image_input)

        for layer_name, feature_map in zip(layer_names, feature_maps):
            num_filters = feature_map.shape[-1]
            size = feature_map.shape[1]

            plt.figure(figsize=(15, 15))
            plt.suptitle(f"Feature Maps from Layer: {layer_name}")

            for i in range(min(6, num_filters)):
                plt.subplot(1, 6, i + 1)
                plt.imshow(feature_map[0, :, :, i], cmap='viridis')
                plt.axis('off')
            
            plt.savefig(f"{self.IMAGE_DIRECTORY}{str_model}_{layer_name}_feature_map.png")
