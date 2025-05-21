from PIL import Image
import numpy as np
import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(src_dir)

# This script is used for testing the accuracy of the models


from src.networks.NND_moreHidden import NNDNetworkV2  # Import neural network class

# Load trained model
nn = NNDNetworkV2(input_size=784, hidden_size1=128, hidden_size2=64, output_size=10)

##nn.load_model("fine_tuned_model.pkl")  # Load the fine-tuned model
# Load the last trained model
nn.load_model("models/last_trained_model.pkl")  # Load the last trained model

# Path to images folder in the project directory
folder_path = "data\images"


# Initialize variables to track correct predictions and total predictions
correct_predictions = 0
total_predictions = 0

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        file_path = os.path.join(folder_path, filename)
        image = Image.open(file_path).convert("L").resize((28, 28))
        input_array = np.array(image)
        input_array = 255 - input_array  # Invert colors
        input_array = input_array / 255.0  # Normalize to [0, 1]
        input_flattened = input_array.flatten().reshape(1, -1)
        prediction = nn.forward_propagation(input_flattened)
        predicted_label = np.argmax(prediction)
        print(f"File: {filename}, Predicted Label: {predicted_label}")
        
        true_label = filename[0]  # Assuming the filename format is 'label_image.png'
        
        
        # number of correct predictions
        if predicted_label == int(true_label):
            correct_predictions += 1
            total_predictions += 1
            
        else:
            total_predictions += 1
        
accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")

