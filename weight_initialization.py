import numpy as np
import matplotlib.pyplot as plt

class ActivationFunc():
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        sig = ActivationFunc.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def ReLU(x):
        return np.maximum(0,x)
    
    @staticmethod
    def RelU_derivative(x):
        return np.where(x > 0, 1, 0)
    
    def __init__(self, func = 'sigmoid'):
        self.func = func
        if func == 'sigmoid':
            self.activate = ActivationFunc.sigmoid
            self.derivative = ActivationFunc.sigmoid_derivative
        elif func == 'ReLU':
            self.activate = ActivationFunc.ReLU
            self.derivative = ActivationFunc.RelU_derivative
        else:
            raise ValueError("Unknown activation function")

def forward_pass(x, hidden_layer_size, weights, activation_function):
    activations = {}
    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]
        a = np.dot(x, weights[i])
        z = activation_function(a)
        activations[i] = z
    return activations

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def initialization(input, node_num):
    if is_number(input):
        return float(input)
    elif input == 'xavier':
        return np.sqrt(1/node_num)
    elif input == 'He':
        return np.sqrt(2/node_num)
    else:
        return 0.1

def main(initializaton_method, activation_function, learning_rate):
    x = np.random.randn(1000,100)
    node_num = 100
    hidden_layer_size = 5
    activations = {}

    weights = {}

    for i in range(hidden_layer_size):
        weights[i] = np.random.randn(node_num, node_num) * initialization(initializaton_method, node_num)

    activations = forward_pass(x, hidden_layer_size, weights, activation_function.activate)

    for i, activation in activations.items():
        plt.subplot(1, len(activations), i+1)
        #plt.yscale('log')
        plt.hist(activation.flatten(), bins=30, range=range(0,2))
        plt.title(f'Layer {i+1}')
    plt.show()

    y_true = np.random.randn(1000, node_num)
    y_pred = activations[hidden_layer_size - 1]

    for i in reversed(range(hidden_layer_size)):
        if i == hidden_layer_size - 1:
            delta = (y_pred - y_true) * activation_function.derivative(activations[i])
        else:
            delta = np.dot(delta, weights[i+1].T) * activation_function.derivative(activations[i])

        if i == 0:
            weight_update = np.dot(x.T, delta)
        else:
            weight_update = np.dot(activations[i - 1].T, delta)

        weights[i] -= learning_rate * weight_update 

    updated_activations = forward_pass(x, hidden_layer_size, weights, activation_function.activate)

    for i, activation in updated_activations.items():
        plt.subplot(1, len(updated_activations), i + 1)
        plt.hist(activation.flatten(), bins=30, range=(0, 2))
        plt.title(f'Layer {i+1} (Updated)')
    plt.show()

initialization_method = input("input sd size or initialization method")
activation_func = input("input activation name(sigmoid or ReLU)")
learning_rate = float(input('input learning rate'))

main(initialization_method, ActivationFunc(activation_func), learning_rate)
