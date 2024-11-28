import numpy as np

def AND_gate(x1, x2):
    # Weight and Threshold
    w1, w2, theta = 0.5, 0.5, 0.7
    output = w1 * x1 + w2 * x2
    return 1 if output > theta else 0

def np_AND_gate():
    x = np.array([0, 1])
    w = np.array([0.5, 0.5])
    b = -0.7
    print(np.sum(w*x) + b)

def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    output = w1 * x1 + w2 * x2
    return 1 if output > theta else 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = -0.7
    y = np.sum(x * w) + b
    return 1 if y > 0 else 0

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND_gate(s1, s2)
    return y


if __name__ == "__main__":
    print(XOR(1,0))