import tensorflow as tf
from keras.layers import TFSMLayer

# Path to the SavedModel directory
model_path = r"C:/Users/Megh/Desktop/2nd yr/extra-curricular/Hackathons/AIML/DL/Potato/models/1"

# Load the SavedModel using TFSMLayer
model = TFSMLayer(model_path, call_endpoint="serving_default")

# Use the model for inference
import numpy as np
# Example input for inference (adjust shape as required by your model)
example_input = np.random.rand(1, 224, 224, 3)  # Modify dimensions based on your model's input shape
predictions = model(example_input)

print(f"Predictions: {predictions}")




