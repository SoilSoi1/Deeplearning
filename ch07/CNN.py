import sys, os


sys.path.append(os.pardir)
from common.optimizer import Adam
from collections import OrderedDict
from dataset.mnist import load_mnist
import numpy as np
from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
import  matplotlib.pyplot as plt

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
        self.test = pool_output_size
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
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # def accuracy(self, x, t):
    #     y = self.predict(x)
    #     compare = (y != t)
    #     acc = np.sum(compare) / len(compare)
    #
    #     return acc

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

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

def draw(data_dic):
    n = len(data_dic)
    pic_num = next(((d, n // d) for d in range(1, int(n ** 0.5) + 1) if n % d == 0), (1, n))
    fig, axs = plt.subplots(pic_num[0], pic_num[1])

    for i, key in zip(range(len(axs)), data_dic.keys()):
        axs[i].plot(data_dic[key])
        axs[i].set_title(f"{key}")

    plt.show()

def plot_loss(data):
    plt.plot(data)

    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()

def train():
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True,flatten=False)

    #log
    train_loss = []
    train_acc = []
    test_acc = []

    # Train params
    lr = 0.1

    epoch = 5

    batch_size = 600
    train_size = x_train.shape[0]
    iter_per_epoch = int(train_size / batch_size)

    iter_num = epoch * iter_per_epoch

    # Init network
    network = SimpleConvNet()

    # Init optimizer
    optimizer = Adam()

    # train
    for iter in range(iter_num):
        # mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        gradient = network.gradient(x_batch, t_batch)

        optimizer.update(network.params, gradient)

        train_loss.append(network.loss(x_batch, t_batch))

        if iter % iter_per_epoch == 0:
            print(f'epoch: {int(iter / 100)}')
            train_acc.append(network.accuracy(x_train, t_train))
            test_acc.append(network.accuracy(x_test, t_test))

    data_set = {"Loss":train_loss, "Train accuracy":train_acc, "Test accuracy":test_acc}
    return data_set
if __name__ == "__main__":
    draw(train())