import numpy as np 
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from src.networks.NND_moreHidden import NNDNetworkV2  # Importing network class


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
"""
  Now we need to flatten the images and normalize the pixel values to be between 0 and 1.
  The images are 28x28 pixels, so we need to reshape them to be 784 pixels (28*28).
    We also need to normalize the pixel values to be between 0 and 1.  
"""

# one hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32')/255 #255 because the pixel values are between 0 and 255.
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32')/255
"""
    Now we need to one-hot encode the labels.
    This means that we need to convert the labels to be a binary matrix representation.
    For example, if the label is 3, we need to convert it to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
    We can do this using the to_categorical function from keras.
"""

"""
    Now feeding the data into the neural network.
    I will be using my own neural network class NNDNetworkV2.
    This class is a simple feedforward neural network with two hidden layers.
    For backpropagation, I will be using the cross entropy loss function.
    and gradient descent for optimization.
    The neural network will be trained for 1000 epochs with a learning rate of 0.001.
"""

input_size = 784  # 28x28 pixels flattened
hidden_size1 = 256
hidden_size2 = 128
output_size = 10  # Number of classes (digits 0-9)
nn = NNDNetworkV2(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=output_size)



patience = 50  # Number of epochs with no improvement after which training will be stopped.
epochs = 200  # Number of epochs for training
# for each epoch the training data is fed into the neural network and the weights are updated.

initial_learning_rate = 0.01  # Initial learning rate
decay_rate = 0.96  # Decay rate for learning rate  # Initialize learning rate
best_loss = float('inf')  # Initialize best loss to infinity

batch_size = 128
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



    

# Evaluate the neural network
print("\nEvaluating the neural network...")
y_test_pred = nn.forward_propagation(X_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)
y_test_true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(y_test_pred_labels == y_test_true_labels) 
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# save the trained model to models folder
model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'last_trained_model.pkl')

nn.save_model(model_path)
    
