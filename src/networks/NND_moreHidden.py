import numpy as np
import pickle

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NNDNetworkV2:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Xavier Initialization
        self.weights_input_hidden1 = np.random.randn(self.input_size, self.hidden_size1) * np.sqrt(2 / (self.input_size+self.hidden_size1))
        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_size1, self.hidden_size2) * np.sqrt(2 / (self.hidden_size1 + self.hidden_size2))
        self.weights_hidden2_output = np.random.randn(self.hidden_size2, self.output_size) * np.sqrt(2/ (self.hidden_size2 + self.output_size))

        self.bias_hidden1 = np.zeros((1, self.hidden_size1))
        self.bias_hidden2 = np.zeros((1, self.hidden_size2))
        self.bias_output = np.zeros((1, self.output_size))
        
    def save_model(self, file_path):
        model_data = {
            "weights_input_hidden1": self.weights_input_hidden1,
            "weights_hidden1_hidden2": self.weights_hidden1_hidden2,
            "weights_hidden2_output": self.weights_hidden2_output,
            "bias_hidden1": self.bias_hidden1,
            "bias_hidden2": self.bias_hidden2,
            "bias_output": self.bias_output
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"Model saved to {file_path}")

    # Load the model from a file
    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
        self.weights_input_hidden1 = model_data["weights_input_hidden1"]
        self.weights_hidden1_hidden2 = model_data["weights_hidden1_hidden2"]
        self.weights_hidden2_output = model_data["weights_hidden2_output"]
        self.bias_hidden1 = model_data["bias_hidden1"]
        self.bias_hidden2 = model_data["bias_hidden2"]
        self.bias_output = model_data["bias_output"]
        print(f"Model loaded from {file_path}")
        

    def forward_propagation(self, x):
        self.hidden_layer1_input = np.dot(x, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_layer1_output = sigmoid(self.hidden_layer1_input)

        self.hidden_layer2_input = np.dot(self.hidden_layer1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden_layer2_output = sigmoid(self.hidden_layer2_input)

        self.final_output = np.dot(self.hidden_layer2_output, self.weights_hidden2_output) + self.bias_output
        return softmax(self.final_output)

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(np.clip(y_pred, 1e-10, 1.0)), axis=1).mean()

    def backward_propagation(self, x, y_true, y_pred, learning_rate=0.01):
        delta_3 = y_pred - y_true
        d_weights_hidden2_output = np.dot(self.hidden_layer2_output.T, delta_3)
        d_bias_output = np.sum(delta_3, axis=0, keepdims=True)

        delta_2 = np.dot(delta_3, self.weights_hidden2_output.T) * sigmoid_derivative(self.hidden_layer2_output)
        d_weights_hidden1_hidden2 = np.dot(self.hidden_layer1_output.T, delta_2)
        d_bias_hidden2 = np.sum(delta_2, axis=0, keepdims=True)

        delta_1 = np.dot(delta_2, self.weights_hidden1_hidden2.T) * sigmoid_derivative(self.hidden_layer1_output)
        d_weights_input_hidden1 = np.dot(x.T, delta_1)
        d_bias_hidden1 = np.sum(delta_1, axis=0, keepdims=True)

        self.weights_hidden2_output -= learning_rate * d_weights_hidden2_output
        self.bias_output -= learning_rate * d_bias_output
        self.weights_hidden1_hidden2 -= learning_rate * d_weights_hidden1_hidden2
        self.bias_hidden2 -= learning_rate * d_bias_hidden2
        self.weights_input_hidden1 -= learning_rate * d_weights_input_hidden1
        self.bias_hidden1 -= learning_rate * d_bias_hidden1