# Neural Network Project

This project showcases the first neural network I built entirely from scratch, without relying on high-level libraries like TensorFlow, PyTorch, or Scikit-learn for the core implementation. The underlying mathematical structure, including forward propagation, backpropagation, with gradient descent of the cross entropy loss function was implemented using NumPy.

## [Demo video](https://youtube.com/shorts/21mqUb5MMSI?feature=share)


External ML libraries were only used for:
- **Dataset Loading**: TensorFlow was used to import the MNIST dataset, and for one-hot encoding the labels.
- **Preprocessing**: Scikit-learn's `train_test_split` was used to split the custom made dataset into sets for training and testing (MNIST from keras comes premade with training and testing splits)

## Results
The initially trained network achieved an accuracy of 98% on the MNIST test.
After finetuning it on a custom made dataset, it achieved an accuracy of 91.45% when tested on this dataset, while still retaining a 93.52% accuracy on the MNIST dataset.

This project demonstrates my understanding of neural networks and machine learning fundamentals. Feel free to explore the code and try it out for yourself!

## Network Architecture

The neural network is a fully connected feedforward network with two hidden layers. The architecture was designed to balance simplicity and performance, making it well-suited for tasks like digit classification.

- **Input Layer**:
  - The input consists of 28x28 grayscale images, flattened into a vector of 784 neurons. Each pixel value is normalized to the range [0, 1].

- **Hidden Layers**:
  1. The first hidden layer contains **256 neurons**.
  2. The second hidden layer contains **128 neurons**.
  - Both layers use sigmoid activation functions to introduce non-linearity and enable the network to learn complex patterns.

- **Output Layer**:
  - The output layer consists of **10 neurons**, each representing one of the digits (0-9). The final predictions are made using a softmax activation function.

This architecture was chosen after experimenting with simpler setups, such as a single hidden layer with fewer neurons. The current design provided satisfactory results without introducing unnecessary complexity.

While adding more hidden layers or neurons could potentially improve performance, the current architecture is sufficient for a task like digit classification. A convolutional neural network (CNN) could achieve better results, but implementing a CNN was beyond the scope of this project, as it focuses on building a neural network manually from scratch.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FirstNeuralNetwork.git
   cd FirstNeuralNetwork
   python -m main

Make sure you have the required packages.

## Features

Fully implemented feedforward neural network from scratch using NumPy.
Supports training, fine-tuning, and testing on both MNIST and custom datasets.
Implements key machine learning concepts:
Forward propagation
Backpropagation
Gradient descent with learning rate decay
Early stopping

## Interactive Drawing UI: 
Draw a digit on the canvas, and the model will predict the digit in real time.
See example below.



## Dependencies

- Python 3.8 or 3.9
- NumPy
- TensorFlow
- Scikit-learn
- Pillow
