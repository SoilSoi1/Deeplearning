import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np

def mean_squared(y, t):
    output = 0.5 * np.sum((y - t))
    return output

def cross_entropy(y, t):
    delta = 1e-7
    output = - np.sum(t * np.log(y + delta))
    return output

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False, one_hot_label=True)

train_size = x_train.shape(0)
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


