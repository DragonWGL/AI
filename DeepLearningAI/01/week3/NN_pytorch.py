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


class NN(torch.nn.Module):

    def __init__(self, in_features, num_hidden_dims):
        super(NN, self).__init__()

        # 初始化weight和bias
        W1 = np.random.randn(num_hidden_dims, in_features)
        B1 = np.zeros(num_hidden_dims)
        W2 = np.random.randn(1, num_hidden_dims)
        B2 = np.zeros(1)
        print('W1 shape %s, B1 shape %s' % (str(W1.shape), str(B1.shape)))
        print('W2 shape %s, B2 shape %s' % (str(W2.shape), str(B2.shape)))

        fc1 = torch.nn.Linear(in_features, num_hidden_dims)
        fc1.weight.data = torch.from_numpy(W1)
        fc1.bias.data = torch.from_numpy(B1)

        fc2 = torch.nn.Linear(num_hidden_dims, 1)
        fc2.weight.data = torch.from_numpy(W2)
        fc2.bias.data = torch.from_numpy(B2)

        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


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
Y = np.where(target == TARGET_LABEL, 1.0, 0.)
print('X shape %s, Y shape %s' % (str(X.shape), str(Y.shape)))

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

NUM_HIDDEN_DIMS = 60
model = NN(X.shape[1], NUM_HIDDEN_DIMS)
optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(1, EPOCHS + 1)):
    # 获取数据
    batch_x, batch_y = torch.from_numpy(train_x), torch.from_numpy(train_y).reshape(-1, 1)

    # 重置求导
    optimizer.zero_grad()

    # 前向传播
    A = model(batch_x)
    loss = torch.nn.functional.binary_cross_entropy(A.squeeze(), batch_y.squeeze())

    # 后向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 1000 == 0:
        print('epoch %5d | loss %.4f' % (epoch, loss.item()))

print('W1 %s' % model.fc1.weight)
print('B1 %s' % model.fc1.bias)
print('W2 %s' % model.fc2.weight)
print('B2 %s' % model.fc2.bias)

batch_test_x, batch_test_y = torch.from_numpy(test_x), test_y.reshape(-1, 1)
model.eval()
A = model(batch_test_x)
Yhat = np.where(A > 0.5, 1, 0)

m = batch_test_x.shape[0]
for i in range(m):
    print('#%3d: predict: %.4f | label: %d' % (i, A[i, 0], batch_test_y[i, 0]))

print('acc %.4f' % ((Yhat == batch_test_y).sum() / m))
