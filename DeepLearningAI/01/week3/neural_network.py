import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("../..")
from activation import *

np.random.seed(1)

LEARNING_RATE = 0.001
EPOCHS = 10000


def train(X, Y, weights, biases, activations):
    # batch size
    m = X.shape[1]

    for epoch in tqdm(range(1, EPOCHS + 1)):
        # 正向传播
        Z_list, A_list = forward(X, weights, biases, activations)

        # 计算成本函数
        J = -(np.dot(Y, np.log(A_list[-1]).transpose()) + np.dot(1 - Y, np.log(1 - A_list[-1]).transpose())) / m
        if epoch % 1000 == 0:
            print('epoch %5d | loss %.4f' % (epoch, J))

        # 反向传播
        dZ2 = A_list[1] - Y  # dJ/dZ2 = dJ/dA2 * dA2/dZ2
        dW2 = np.dot(dZ2, A_list[0].transpose()) / m  # dJ/dW2 = dJ/dZ2 * dZ2/dW2
        dB2 = np.sum(dZ2, axis=1, keepdims=True) / m  # dJ/dB2 = dJ/dZ2 * dZ2/dB2
        # dZ2/dA1 = W2
        # dA1/dZ1 = drelu(Z1)/dZ1 = np.where(A1 > 0, 1, 0)
        dZ1 = np.multiply(np.dot(weights[1].transpose(), dZ2),
                          np.where(Z_list[0] > 0, 1, 0))  # dJ/dZ1 = dJ/dZ2 * dZ2/dA1 * dA1/dZ1
        dW1 = np.dot(dZ1, X.transpose()) / m  # dJ/dW1 = dJ/dZ1 * dZ1/dW1
        dB1 = np.sum(dZ1, axis=1, keepdims=True) / m  # dJ/dB1 = dJ/dZ1 * dZ1/dB1

        # 更新参数
        weights[1] -= LEARNING_RATE * dW2
        biases[1] -= LEARNING_RATE * dB2
        weights[0] -= LEARNING_RATE * dW1
        biases[0] -= LEARNING_RATE * dB1


def forward(X, weights, biases, activations):
    Z_list = []
    A_list = []

    for i in range(len(weights)):
        Z_list.append(np.dot(weights[i], X if len(A_list) == 0 else A_list[-1]) + biases[i])
        A_list.append(activation(activations[i], Z_list[-1]))

    return Z_list, A_list


def test(X, Y, weights, baises, activations):
    _, A_list = forward(X, weights, baises, activations)
    A = A_list[-1]
    y_hat = np.where(A > 0.5, 1, 0)

    m = X.shape[1]
    for i in range(m):
        print('#%3d: predict: %.4f | label: %d' % (i, A[0, i], Y[0, i]))

    print('acc %.4f' % ((y_hat == Y).sum() / m))


if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    print(iris.keys())

    # 加载特征和标签
    data = iris['data']
    target = iris['target']
    print('data shape %s, target shape %s' % (str(data.shape), str(target.shape)))

    # 转换成二元逻辑回归数据
    TARGET_LABEL = 1  # {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    X = data
    Y = np.where(target == TARGET_LABEL, 1, 0)
    print('X shape %s, Y shape %s' % (str(X.shape), str(Y.shape)))

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

    # 初始化weight和bias
    NUM_HIDDEN_DIMS = 60
    W1 = np.random.randn(NUM_HIDDEN_DIMS, X.shape[1])
    B1 = np.zeros((NUM_HIDDEN_DIMS, 1))
    W2 = np.random.randn(1, NUM_HIDDEN_DIMS)
    B2 = np.zeros((1, 1))
    print('W1 shape %s, B1 shape %s' % (str(W1.shape), str(B1.shape)))
    print('W2 shape %s, B2 shape %s' % (str(W2.shape), str(B2.shape)))

    weights = [W1, W2]
    biases = [B1, B2]
    activations = ['relu', 'sigmoid']

    assert len(weights) == len(biases) == len(activations)

    train(train_x.transpose(), train_y.reshape(1, -1), weights, biases, activations)

    print('W1 %s' % W1)
    print('B1 %s' % B1)
    print('W2 %s' % W2)
    print('B2 %s' % B2)

    test(test_x.transpose(), test_y.reshape(1, -1), weights, biases, activations)
