import os
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import mnist
from src.networks.NND_moreHidden import NNDNetworkV2

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Preprocess the data
x_test = x_test.reshape(-1, 784) / 255.0  # Flatten and normalize the images

# Load your fine-tuned model
nn = NNDNetworkV2(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10)
nn.load_model("models/finetuned_model.pkl")  # Load the fine-tuned model
#nn.load_model("models/last_trained_model.pkl")  # Load the last trained model

# Test the model
predictions = nn.forward_propagation(x_test)  # Get predictions for all samples
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

# Calculate accuracy
accuracy = np.mean(predicted_labels == y_test) * 100
print(f"Accuracy on the entire MNIST dataset: {accuracy:.2f}%")