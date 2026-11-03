import os
import numpy as np
from PIL import Image
from src.networks.NND_moreHidden import NNDNetworkV2
from src.data import load_mnist, preprocess_digit

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = load_mnist()



# Preprocess the data
x_test = x_test.reshape(-1, 784) / 255.0  # Flatten and normalize the images

# Load the trained model
nn = NNDNetworkV2(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10)
nn.load_model("models/digit_recog.pkl")

# Test the model
predictions = nn.forward_propagation(x_test)  # Get predictions for all samples
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

# Calculate accuracy
accuracy = np.mean(predicted_labels == y_test) * 100
print(f"Accuracy on the entire MNIST dataset: {accuracy:.2f}%")

# Diagnostic only: these drawings do not affect training or model selection.
drawing_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
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
