import numpy as np


def activation(key, Z):
    if key == 'relu':
        return relu(Z)
    if key == 'lrelu':
        return leaky_relu(Z)
    elif key == 'tanh':
        return tanh(Z)
    elif key == 'sigmoid':
        return sigmoid(Z)
    elif key == 'softmax':
        return softmax(Z)
    else:
        raise Exception('unsupported activation %s' % key)


def derivative(key, Z, A):
    if key == 'relu':
        return drelu(Z)
    if key == 'lrelu':
        return dleaky_relu(Z)
    elif key == 'tanh':
        return dtanh(A)
    elif key == 'sigmoid':
        return dsigmoid(A)
    elif key == 'softmax':
        return dsoftmax(A)
    else:
        raise Exception('unsupported derivative %s' % key)


def relu(Z):
    return np.maximum(Z, np.zeros(Z.shape))


def leaky_relu(Z, negative_slope=0.01):
    return np.maximum(Z, Z * negative_slope)


def tanh(Z):
    return np.tanh(Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    exp_Z = np.exp(Z)
    sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)
    return exp_Z / sum_exp_Z


def drelu(Z):
    return np.where(Z > 0, 1, 0)


def dleaky_relu(Z, negative_slope=0.01):
    return np.where(Z > 0, 1, negative_slope)


def dtanh(A):
    return 1 - np.multiply(A, A)


def dsigmoid(A):
    return np.multiply(A, (1 - A))


def dsoftmax(A):
    return None
