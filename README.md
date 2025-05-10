# Neural Network Project

This project showcases the first neural network I built entirely from scratch, without relying on high-level libraries like TensorFlow, PyTorch, or Scikit-learn for the core implementation. The underlying mathematical structure, including forward propagation, backpropagation, and gradient descent, was implemented using NumPy.

## [Demo video](https://youtube.com/shorts/21mqUb5MMSI?feature=share)


External ML libraries were only used for:
- **Dataset Loading**: TensorFlow was used to import the MNIST dataset, and for one-hot encoding the labels.
- **Preprocessing**: Scikit-learn's `train_test_split` was used to split the custom made dataset into sets for training and testing (MNIST from keras comes premade with training and testing splits)

## Results
The initially trained network achieved an accuracy of 98% on the MNIST test.
After finetuning it on a custom made dataset, it achieved an accuracy of 91.45% when tested on this dataset, while still retaining a 93.52% accuracy on the MNIST dataset.

This project demonstrates my understanding of neural networks and machine learning fundamentals. Feel free to explore the code and try it out for yourself!

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
