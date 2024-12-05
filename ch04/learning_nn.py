import sys, os

import numpy as np
import matplotlib.pyplot as plt

from common.layers import Affine, Relu, SoftmaxWithLoss

sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

from collections import OrderedDict
from dataset.mnist import load_mnist

class TwoLayersNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # Initial weight
        self.params = {}
        'why times 0.01 ? '
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        # Generate layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # 注释掉的为手动实现的计算
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # z2 = sigmoid(a2)

        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 数值微分法求梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 反向传播法求梯度（链式法则）
    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 向前传播方向的层
        layers = list(self.layers.values())
        # BP时需要反方向
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

# Usage: optimizer = SGD()
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        # 初始化速度，这里的速度也为字典类型，值全为0
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


def plot_loss(data):
    plt.plot(data)

    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()

def train():
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

    # Log
    train_loss_list_float = []
    train_acc_list = []
    test_acc_list = []



    # 超参数
    iters_num = 1000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    # Steps every epoch involves
    iter_per_epoch = max(train_size / batch_size, 1)

    # 初始化神经网络
    network = TwoLayersNet(input_size=784, hidden_size=50, output_size=10)
    optimizer = Momentum()

    for i in range(iters_num):
        print(i)
        #获取mini_batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.gradient(x_batch, t_batch)

        # 更新参数
        optimizer.update(network.params, grad)

        # Log the period
        loss = network.loss(x_batch, t_batch)
        train_loss_list_float.append(loss)

        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            # print(f'''train accuracy | test accuracy
            # {train_acc} | {test_acc}
            # ''')

    # 可视化
    train_loss_list = [f.item() for f in train_loss_list_float]
    plot_loss(train_loss_list)

if __name__ == "__main__":
    train()