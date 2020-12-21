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
            print('epoch %5d | loss %f' % (epoch, J))

        # 反向传播
        dYhat = -Y / A_list[-1] + (1 - Y) / (1 - A_list[-1])  # dJ/dYhat = -Y / Yhat + (1 - Y) / (1 - Yhat)
        backward(X, weights, biases, activations, Z_list, A_list, dYhat)


def forward(X, weights, biases, activations):
    Z_list = []
    A_list = []

    for i in range(len(weights)):
        Z_list.append(np.dot(weights[i], X if len(A_list) == 0 else A_list[i - 1]) + biases[i])
        A_list.append(activation(activations[i], Z_list[i]))

    return Z_list, A_list


def backward(X, weights, biases, activations, Z_list, A_list, dYhat):
    m = X.shape[1]

    dZ_List = []
    dW_list = []
    dB_list = []

    for i in range(len(activations) - 1, -1, -1):
        dA_dZ = derivative(activations[i], Z_list[i], A_list[i])
        dZ_List.insert(0,
                       np.multiply(dYhat if len(dZ_List) == 0 else np.dot(weights[i + 1].transpose(), dZ_List[0]),
                                   dA_dZ))
        dW_list.insert(0, np.dot(dZ_List[0], A_list[i - 1].transpose() if i > 0 else X.transpose()) / m)
        dB_list.insert(0, np.sum(dZ_List[0], axis=1, keepdims=True) / m)

        weights[i] -= LEARNING_RATE * dW_list[0]
        biases[i] -= LEARNING_RATE * dB_list[0]

    return weights, biases


def test(X, Y, W_list, B_list, G_list):
    _, A_list = forward(X, W_list, B_list, G_list)
    A = A_list[-1]
    Yhat = np.where(A > 0.5, 1, 0)

    m = X.shape[1]
    for i in range(m):
        print('#%3d: predict: %f | label: %d' % (i, A[0, i], Y[0, i]))

    print('acc %f' % ((Yhat == Y).sum() / m))


if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    print(iris.keys())

    # 加载特征和标签
    data = iris['data']
    target = iris['target']
    print('data shape %s, target shape %s' % (str(data.shape), str(target.shape)))

    # 转换成二元逻辑回归数据
    TARGET_LABEL = 2  # {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    X = data
    Y = np.where(target == TARGET_LABEL, 1, 0)
    print('X shape %s, Y shape %s' % (str(X.shape), str(Y.shape)))

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

    # 初始化网络
    # 隐藏节点列表
    HIDDEN_DIMS = [50, 50, 30, 1]
    # 激活函数列表
    ACTIVATIONS = ['relu', 'lrelu', 'tanh', 'sigmoid']

    assert len(HIDDEN_DIMS) == len(ACTIVATIONS)

    # 初始化weight和bias
    weights = []
    biases = []
    for i in range(len(HIDDEN_DIMS)):
        weights.append(np.random.randn(HIDDEN_DIMS[i], X.shape[1] if len(weights) == 0 else weights[-1].shape[0]))
        biases.append(np.zeros((HIDDEN_DIMS[i], 1)))
        print('weight %d shape %s, bias %d shape %s' % (i + 1, str(weights[i].shape), i + 1, str(biases[i].shape)))

    train(train_x.transpose(), train_y.reshape(1, -1), weights, biases, ACTIVATIONS)

    for i in range(len(weights)):
        print('W%d %s' % (i + 1, weights[i]))
        print('B%d %s' % (i + 1, biases[i]))

    test(test_x.transpose(), test_y.reshape(1, -1), weights, biases, ACTIVATIONS)
