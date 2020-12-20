import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("../..")
from activation import *

np.random.seed(1)

LEARNING_RATE = 0.1
EPOCHS = 10000


def train(X, Y, W, B):
    # batch size
    m = X.shape[1]

    for epoch in tqdm(range(1, EPOCHS + 1)):
        # 正向传播
        A = forward(X, W, B)

        # 计算成本函数
        # J = -∑(Y * log(Yhat) + (1-Y) * log(1 - Yhat)) / m
        J = -(np.dot(Y, np.log(A).transpose()) + np.dot(1 - Y, np.log(1 - A).transpose())) / m
        if epoch % 1000 == 0:
            print('epoch %5d | loss %.4f' % (epoch, J))

        # 反向传播
        # dJ/dA = -Y / A + (1 - Y) / (1 - A)
        # dA/dZ = dsigmod(Z)/dZ = A * (1 - A)
        dZ = A - Y  # dJ/dZ = dJ/dA * dA/dZ
        dW = np.dot(X, dZ.transpose()) / m  # ∑(dJ/dW) / m = ∑(dJ/dZ * dZ/dW) / m
        dB = np.sum(dZ, axis=1, keepdims=True) / m  # ∑(dJ/dB) / m = ∑(dJ/dZ * dZ/dB) / m

        # 更新参数
        W -= LEARNING_RATE * dW
        B -= LEARNING_RATE * dB


def forward(X, W, B):
    Z = np.dot(W.transpose(), X) + B
    A = sigmoid(Z)

    return A


def test(X, Y, W, B):
    A = forward(X, W, B)
    Yhat = np.where(A > 0.5, 1, 0)

    m = X.shape[1]
    for i in range(m):
        print('#%3d: predict: %.4f | label: %d' % (i, A[0, i], Y[0, i]))

    print('acc %.4f' % ((Yhat == Y).sum() / m))


if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    print(iris.keys())

    # 加载特征和标签
    data = iris['data']
    target = iris['target']
    print('data shape %s, target shape %s' % (str(data.shape), str(target.shape)))

    # 转换成二元逻辑回归数据
    TARGET_LABEL = 0  # {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    X = (np.vstack((data[0:50], data[50:100])))
    Y = np.where(np.hstack((target[0:50], target[50:100])) == TARGET_LABEL, 1, 0)
    print('X shape %s, Y shape %s' % (str(X.shape), str(Y.shape)))

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

    # 初始化weight和bias
    W = np.zeros((X.shape[1], 1))
    B = np.zeros((1, 1))
    print('W shape %s, B shape %s' % (str(W.shape), str(B.shape)))

    train(train_x.transpose(), train_y.reshape(1, -1), W, B)

    print('weights %s' % W)
    print('baises %s' % B)

    test(test_x.transpose(), test_y.reshape(1, -1), W, B)
