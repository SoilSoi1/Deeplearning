import sys, os

import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

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

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)

        return z2

    # t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

def plot_loss(data):
    plt.plot(data, marker = 'o')

    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()

(x_train, t_train), (x_test, t_test) = \
load_mnist(normalize=True, one_hot_label=True)

train_loss_list_float = []

# 超参数
iters_num = 2
epoch = iters_num
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 初始化神经网络
network = TwoLayersNet(input_size=784, hidden_size=50, output_size=10)

for i in range(epoch):
    print(i)
    #获取mini_batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Log the period
    loss = network.loss(x_train, t_train)
    train_loss_list_float.append(loss)

train_loss_list = [f.item() for f in train_loss_list_float]
print(train_loss_list)
plot_loss(train_loss_list)
