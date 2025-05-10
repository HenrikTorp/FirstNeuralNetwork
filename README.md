# Neural Network Project

## Table of Contents
- [Introduction](#Introduction)
- [Demo Video](#demo-video)
- [Results](#results)
- [How to Run](#how-to-run)
- [Network Architecture](#network-architecture)
- [Features](#features)
- [Interactive Drawing UI](#interactive-drawing-ui)
- [Dependencies](#dependencies)


## Introduction
This project showcases the first neural network I built entirely from scratch, without relying on high-level libraries like TensorFlow, PyTorch, or Scikit-learn for the core implementation. The underlying mathematical structure, including forward propagation, backpropagation, with gradient descent of the cross entropy loss function was implemented using NumPy.

## [Demo video](https://youtube.com/shorts/21mqUb5MMSI?feature=share)

Draw a digit on the canvas, and the model will predict the digit in real time. 

For the best results, draw the digit **within the rectangle** provided in the UI. This rectangle helps guide users to draw digits in the correct area for accurate predictions.

Example:
![Drawing UI Example](https://raw.githubusercontent.com/your-username/your-repo/main/assets/digitNN.png))


External ML libraries were only used for:
- **Dataset Loading**: TensorFlow was used to import the MNIST dataset, and for one-hot encoding the labels.
- **Preprocessing**: Scikit-learn's `train_test_split` was used to split the custom made dataset into sets for training and testing (MNIST from keras comes premade with training and testing splits)

## Results
After training on the MNIST dataset, the neural network achieved an accuracy of 98% when tested on digits from the dataset it had not yet seen.
After finetuning it on a custom made dataset, it achieved an accuracy of 91.45% when tested on this dataset, while still retaining a 93.52% accuracy on the MNIST dataset.

There are limitations, however. The neural network performs best when digits are drawn cleanly and likely shows a bias toward my drawing style. And try to draw the digits within the 
If you want to try it out for yourself, see installation guide below!


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FirstNeuralNetwork.git
   cd FirstNeuralNetwork
   python -m main

Ensure the required dependencies are installed before running the project.



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


## Features

- Fully implemented feedforward neural network from scratch using NumPy.
- Supports training, fine-tuning, and testing on both MNIST and custom datasets.
- Implements key machine learning concepts:
  - Forward propagation
  - Backpropagation
  - Gradient descent with learning rate decay
  - Early stopping

## Interactive Drawing UI: 
Draw a digit on the canvas, and the model will predict the digit in real time.
See example below.



## Dependencies

- Python 3.8 or 3.9 (compatibility with tensorflow)
- NumPy
- TensorFlow
- Scikit-learn
- Pillow


## Conclusion
I had fun making this project. It has its limitations, but it greatly increased my practical
understanding of how neural networks work, and understanding its limitations in itself was very valuable. In future projects, I am excited to experiment with more sophisticated architectures, such as convolutional neural networks (CNNs).
