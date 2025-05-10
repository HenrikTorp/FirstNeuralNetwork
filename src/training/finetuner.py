import os
from PIL import Image
import numpy as np
import os 
# Disable oneDNN optimization to avoid warning messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# testplit from scikit-learn and using keras for one-hot encoding
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.networks.NND_moreHidden import NNDNetworkV2  # Import your neural network class

def load_data(data_dir):
    """
    Load images and labels from the specified directory.
    The images should be in PNG or JPG format, and the labels should be the first character of the filename.
    """
    images = []
    labels = []

    for img_file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_file)
        if img_file.endswith('.png') or img_file.endswith('.jpg'):
            try:
                # Open the image, convert to grayscale, and resize to 28x28
                img = Image.open(img_path).convert('L').resize((28, 28))
                img_array = np.array(img).flatten() / 255.0  # Normalize pixel values to [0, 1]
                img_array = 1.0 - img_array  # Invert colors
                images.append(img_array)

                
                label = int(img_file[0])
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    # Convert lists to NumPy arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y

# Load the data
X, y = load_data('data/images')

# Split the data into training and test sets
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train, y_test = keras.utils.to_categorical(y_train, num_classes=10), keras.utils.to_categorical(y_test, num_classes=10)

nn = NNDNetworkV2(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10)  # Initialize the neural network
nn.load_model('models/last_trained_model.pkl')
patience = 50  # Number of epochs with no improvement after which training will be stopped.
epochs = 200  # Number of epochs for training
# for each epoch the training data is fed into the neural network and the weights are updated.

initial_learning_rate = 0.0001  # Initial learning rate
decay_rate = 0.96  # Decay rate for learning rate  # Initialize learning rate
best_loss = float('inf')  # Initialize best loss to infinity

batch_size = 16
num_batches = X_train.shape[0] // batch_size

# Train the neural network
print("Training the neural network...")
for epoch in range(epochs):
    for i in range(num_batches):
        # Get the batch data
        X_batch = X_train[i * batch_size:(i + 1) * batch_size]
        y_batch = y_train[i * batch_size:(i + 1) * batch_size]
        
        
        # Forward propagation
        y_pred = nn.forward_propagation(X_batch)
        
        # Compute the loss
        loss = nn.cross_entropy_loss(y_batch, y_pred)
        
        # Backward propagation
        learning_rate = initial_learning_rate * (decay_rate ** (epoch // 10))  # Decay learning rate every 10 epochs
        nn.backward_propagation(X_batch, y_batch, y_pred, learning_rate)
    
    
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    # Early stopping
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


# check accuracy on the test set in %
y_test_pred = nn.forward_propagation(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1) # Convert probabilities to class labels
y_test_true_labels = np.argmax(y_test, axis=1) # Convert one-hot encoded labels to class labels
accuracy = np.mean(y_test_pred_labels == y_test_true_labels) * 100

print(f"Accuracy on the test set: {accuracy:.2f}%")



finetune_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'finetuned_model.pkl')
nn.save_model(finetune_model_path)  # Save the trained model to the specified path