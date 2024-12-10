from collections import OrderedDict

import numpy as np
from torch.ao.nn.quantized import Softmax

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss


class SimpleConvNet:
    def __init__(self, input_dim=(1 ,28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size = 100, output_size = 10, weight_init = 0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        # For the first step, get the output size
        conv_output_size = int(input_size - filter_size + 2*filter_pad) / filter_stride +1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # Set params
        self.params = {}
        self.params['W1'] = weight_init * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init * np.random.randn(
            pool_output_size, hidden_size
        )
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init * np.random.randn(
            hidden_size, output_size
        )
        self.params['b3'] = np.zeros(output_size)

        # Set layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(
            self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad']
        )
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(2,2,2,0)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        # Last layer for learning, not for inference
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.value():
            y = layer.forward(x)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad = {}
        grad['W1'] = self.layers['Conv1'].dW
        grad['b1'] = self.layers['Conv1'].db
        grad['W2'] = self.layers['Affine1'].dW
        grad['b2'] = self.layers['Affine1'].db
        grad['W3'] = self.layers['Affine2'].dW
        grad['b3'] = self.layers['Affine2'].db

        return grad

def train():
