import numpy as np 
import os 
import sys
import json
from inspect import signature
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from PIL import Image
from src.data import load_mnist, preprocess_digit, augment_batch
from src.networks.NND_moreHidden import NNDNetworkV2  # Importing network class
from src.training.plots import plot_training


def evaluate_epoch(nn, images, labels, batch_size):
    """Measure loss and accuracy without updating the epoch's final weights."""
    total_loss = 0.0
    correct = 0
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        scores = nn.forward_propagation(batch_images)
        batch_loss = nn.cross_entropy_loss(batch_labels, nn.final_output)
        if not np.isfinite(batch_loss):
            raise RuntimeError("Evaluation loss is not finite; training stopped.")
        total_loss += batch_loss * len(batch_images)
        correct += np.count_nonzero(
            np.argmax(scores, axis=1) == np.argmax(batch_labels, axis=1)
        )
    return total_loss / len(images), correct / len(images)


training_seed = 42
np.random.seed(training_seed)

# Preserve the supplied checkpoint and any previous completed run.
model_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models',
    'digit_recog.pkl'
)
if os.path.exists(model_path):
    raise FileExistsError("Choose a new model filename before starting training.")

(X_train, y_train), (X_test, y_test) = load_mnist()
"""
  Now we need to flatten the images and normalize the pixel values to be between 0 and 1.
  The images are 28x28 pixels, so we need to reshape them to be 784 pixels (28*28).
    We also need to normalize the pixel values to be between 0 and 1.  
"""

# one hot encode the labels
y_train = np.eye(10, dtype=np.float32)[y_train]
y_test = np.eye(10, dtype=np.float32)[y_test]

X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32')/255 #255 because the pixel values are between 0 and 255.
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32')/255

# Keep validation examples separate from training and the final test set.
validation_size = 5000
split_rng = np.random.default_rng(42)
indices = split_rng.permutation(X_train.shape[0])
validation_indices = indices[:validation_size]
training_indices = indices[validation_size:]
X_validation, y_validation = X_train[validation_indices], y_train[validation_indices]
X_train, y_train = X_train[training_indices], y_train[training_indices]
"""
    Now we need to one-hot encode the labels.
    This means that we need to convert the labels to be a binary matrix representation.
    For example, if the label is 3, we need to convert it to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
    We can do this by selecting the label rows from a NumPy identity matrix.
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



patience = 30  # Number of epochs with no improvement after which training will be stopped.
epochs = 2000  # Number of epochs for training
# for each epoch the training data is fed into the neural network and the weights are updated.

initial_learning_rate = 0.001  # Initial learning rate
decay_rate = 0.96  # Decay rate for learning rate  # Initialize learning rate
best_loss = float('inf')  # Initialize best loss to infinity
patience_counter = 0
best_weights = None
best_epoch = 0
parameter_names = (
    "weights_input_hidden1", "weights_hidden1_hidden2", "weights_hidden2_output",
    "bias_hidden1", "bias_hidden2", "bias_output"
)
history = {
    "train_loss": [], "validation_loss": [],
    "train_accuracy": [], "validation_accuracy": []
}

augment_rng = np.random.default_rng(training_seed)
batch_size = 128 # this determines how many digits are fed into the neural network at once.

history_path = os.path.splitext(model_path)[0] + "_history.json"
history_temp_path = history_path + ".tmp"
run = {
    "history": history,
    "settings": {
        "architecture": [input_size, hidden_size1, hidden_size2, output_size],
        "training_seed": training_seed,
        "validation_seed": 42,
        "validation_size": validation_size,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "optimizer": {"name": "Adam", "beta1": nn.beta1, "beta2": nn.beta2, "epsilon": nn.epsilon},
        "learning_rate": {"initial": initial_learning_rate, "decay": decay_rate, "decay_every_epochs": 10},
        "augmentation": {
            name: parameter.default
            for name, parameter in signature(augment_batch).parameters.items()
            if name not in ("images", "rng")
        },
    },
}

# Train the neural network
print("Training the neural network...")
for epoch in range(epochs):
    indices = np.random.permutation(X_train.shape[0])
    for start in range(0, X_train.shape[0], batch_size):
        # Use the same shuffled indices for images and labels.
        batch_indices = indices[start:start + batch_size]
        # Fresh random shifts and noise every time a batch is drawn; training data only.
        X_batch = augment_batch(X_train[batch_indices], augment_rng)
        y_batch = y_train[batch_indices]
        
        
        # Forward propagation
        y_pred = nn.forward_propagation(X_batch)
        
        # Compute the loss
        loss = nn.cross_entropy_loss(y_batch, nn.final_output)
        
        # Backward propagation
        learning_rate = initial_learning_rate * (decay_rate ** (epoch // 10))  # Decay learning rate every 10 epochs
        nn.backward_propagation(X_batch, y_batch, y_pred, learning_rate)
    
    
    
    # Stop a failed run before it can be evaluated or saved.
    if not all(np.isfinite(getattr(nn, name)).all() for name in parameter_names):
        raise RuntimeError("Model parameters are not finite; training stopped.")

    # Both curves use the same weights, after the epoch's final update.
    train_loss, train_accuracy = evaluate_epoch(
        nn, X_train, y_train, batch_size
    )
    validation_loss, validation_accuracy = evaluate_epoch(
        nn, X_validation, y_validation, batch_size
    )
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["train_accuracy"].append(train_accuracy)
    history["validation_accuracy"].append(validation_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Training loss: {train_loss:.4f}, "
        f"Validation loss: {validation_loss:.4f}"
    )

    if validation_loss < best_loss:
        best_loss = validation_loss
        best_epoch = epoch + 1
        best_weights = {
            name: getattr(nn, name).copy() for name in parameter_names
        }
        patience_counter = 0
    else:
        patience_counter += 1

    run.update(
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        completed_epochs=len(history["train_loss"]),
    )
    with open(history_temp_path, "w", encoding="utf-8") as file:
        json.dump(run, file, indent=2, allow_nan=False)
    os.replace(history_temp_path, history_path)

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

if best_weights is None:
    raise RuntimeError("No valid training epoch is available for evaluation.")

# Restore the best validation weights before the final test and save.
for name, value in best_weights.items():
    setattr(nn, name, value)
print(f"Selected epoch: {best_epoch}, Validation loss: {best_loss:.4f}")

# Evaluate the neural network
print("\nEvaluating the neural network...")
y_test_pred = nn.forward_propagation(X_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)
y_test_true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(y_test_pred_labels == y_test_true_labels) 
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# save the trained model to models folder

nn.save_model(model_path)

# Diagnostic only: these drawings do not affect training or model selection.
drawing_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'images')
correct = total = skipped = 0
try:
    drawing_files = os.listdir(drawing_folder)
except OSError:
    drawing_files = None

if drawing_files is None:
    print("Saved-drawing diagnostic: unavailable (data/images could not be read).")
else:
    for filename in sorted(drawing_files):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if filename[0] not in "0123456789":
            skipped += 1
            continue
        try:
            with Image.open(os.path.join(drawing_folder, filename)) as image:
                input_array = preprocess_digit(image)
        except (OSError, ValueError):
            skipped += 1
            continue
        if input_array is None:
            skipped += 1
            continue
        prediction = nn.forward_propagation(input_array.reshape(1, -1))
        correct += int(np.argmax(prediction) == int(filename[0]))
        total += 1

    if total:
        print(f"Saved-drawing diagnostic: {correct}/{total} correct ({100 * correct / total:.2f}%), {skipped} skipped.")
    else:
        print(f"Saved-drawing diagnostic: no valid labeled images ({skipped} skipped).")

# Save beside the separately named checkpoint, after the existing diagnostics.
plot_path = os.path.splitext(model_path)[0] + "_training.png"
plot_training(history, best_epoch, plot_path)
print(f"Training plot saved to {plot_path}")
