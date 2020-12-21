import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn
import torch.nn.functional
from torch.optim import SGD
from tqdm import tqdm

np.random.seed(1)
torch.manual_seed(1)

LEARNING_RATE = 0.001
EPOCHS = 10000


class DNN(torch.nn.Module):

    def __init__(self, in_features, hidden_dims, activations):
        super(DNN, self).__init__()

        sequential = torch.nn.Sequential()

        # 初始化weight和bias
        for i in range(len(hidden_dims)):
            if i == 0:
                W = np.random.randn(hidden_dims[i], in_features)
            else:
                W = np.random.randn(hidden_dims[i], hidden_dims[i - 1])
            B = np.zeros(hidden_dims[i])
            print('W%d shape %s, B%d shape %s' % (i + 1, str(W.shape), i + 1, str(B.shape)))

            fc = torch.nn.Linear(W.shape[1], W.shape[0])
            fc.weight.data = torch.from_numpy(W)
            fc.bias.data = torch.from_numpy(B)

            sequential.add_module('fc%d' % (i + 1), fc)
            if activations[i] == 'relu':
                sequential.add_module('activation%d' % (i + 1), torch.nn.ReLU())
            elif activations[i] == 'lrelu':
                sequential.add_module('activation%d' % (i + 1), torch.nn.LeakyReLU())
            elif activations[i] == 'tanh':
                sequential.add_module('activation%d' % (i + 1), torch.nn.Tanh())
            elif activations[i] == 'sigmoid':
                sequential.add_module('activation%d' % (i + 1), torch.nn.Sigmoid())

        self.sequential = sequential

    def forward(self, x):
        x = self.sequential(x)

        return x


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
Y = np.where(target == TARGET_LABEL, 1.0, 0.)
print('X shape %s, Y shape %s' % (str(X.shape), str(Y.shape)))

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

# 初始化weight和bias
# 隐藏节点列表
HIDDEN_DIMS = [50, 50, 30, 1]
# 激活函数列表
ACTIVATIONS = ['relu', 'lrelu', 'tanh', 'sigmoid']

assert len(HIDDEN_DIMS) == len(ACTIVATIONS)

model = DNN(X.shape[1], HIDDEN_DIMS, ACTIVATIONS)
optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(1, EPOCHS + 1)):
    # 获取数据
    batch_x, batch_y = torch.from_numpy(train_x), torch.from_numpy(train_y).reshape(-1)

    # 重置求导
    optimizer.zero_grad()

    # 前向传播
    A = model(batch_x)

    loss = torch.nn.functional.binary_cross_entropy(A.squeeze(), batch_y)

    # 后向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 1000 == 0:
        print('epoch %5d | loss %.4f' % (epoch, loss.item()))

for i in range(len(model.sequential)):
    if type(model.sequential[i]) == torch.nn.Linear:
        print('W%i %s' % (i + 1, model.sequential[i].weight))
        print('B%i %s' % (i + 1, model.sequential[i].bias))

batch_test_x, batch_test_y = torch.from_numpy(test_x), test_y.reshape(-1, 1)
A = model(batch_test_x)
Yhat = np.where(A > 0.5, 1, 0)

m = batch_test_x.shape[0]
for i in range(m):
    print('#%3d: predict: %f | label: %d' % (i, A[i, 0], batch_test_y[i, 0]))

print('acc %f' % ((Yhat == batch_test_y).sum() / m))
