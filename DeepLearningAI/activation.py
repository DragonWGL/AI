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


def drelu(Z):
    return np.where(Z > 0, 1, 0)


def dleaky_relu(Z, negative_slope=0.01):
    return np.where(Z > 0, 1, negative_slope)


def dtanh(A):
    return 1 - np.multiply(A, A)


def dsigmoid(A):
    return np.multiply(A, (1 - A))
