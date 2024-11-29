from array import array

import numpy as np

# Multi-dimensional array
def multidim_array():

    # Create arrays
    A = np.array([1, 2, 3, 4])
    B = np.array([[1,2], [3,4], [5,6]])

    # Output the dimension number
    print(np.ndim(A))
    print(str(np.ndim(B)) + '|' + str(B.shape))

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def identify_function(x):
    return x

def softmax(x:array):
    c = np.max(x)
    input_sum = np.sum(np.exp(x - c))
    y = np.exp(x - c) / input_sum
    return y

# A = X * W + B
def neural_2322():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5],
                   [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    # Output of layer 1
    Z1 = sigmoid(np.dot(X, W1) + B1)

    W2 = np.array([[0.1, 0.4],
                   [0.2, 0.5],
                   [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    # Output of layer2
    Z2 = sigmoid(np.dot(Z1, W2) + B2)

    W3 = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    B3 = np.array([0.1, 0.2])

    Y = identify_function(np.dot(Z2, W3) + B3) # Or Y = np.dot(Z2, W3) + B3

    return Y

def softmax_1():
    a = np.array([1000, 900, 990])

    return softmax(a)

if __name__ == "__main__":
    print(np.sum(softmax_1()))