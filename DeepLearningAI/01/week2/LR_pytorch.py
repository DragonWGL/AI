import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn
import torch.nn.functional
from torch.optim import SGD
from tqdm import tqdm

np.random.seed(1)
torch.manual_seed(1)

LEARNING_RATE = 0.1
EPOCHS = 10000


class LR(torch.nn.Module):

    def __init__(self, in_features):
        super(LR, self).__init__()

        # 初始化weight和bias
        W = np.zeros((in_features, 1))
        B = np.zeros((1, 1))
        print('W shape %s, B shape %s' % (str(W.shape), str(B.shape)))

        linear = torch.nn.Linear(in_features, 1)
        linear.weight.data = torch.from_numpy(W.transpose())
        linear.bias.data = torch.from_numpy(B)

        self.linear = linear

    def forward(self, x):
        x = self.linear(x)
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
TARGET_LABEL = 0  # {'setosa': 0, 'versicolor': 1, 'virginica': 2}
X = (np.vstack((data[0:50], data[50:100])))
Y = np.where(np.hstack((target[0:50], target[50:100])) == TARGET_LABEL, 1.0, 0.)
print('X shape %s, Y shape %s' % (str(X.shape), str(Y.shape)))

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

model = LR(X.shape[1])
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

print('weights %s' % model.linear.weight)
print('biases %s' % model.linear.bias)

batch_test_x, batch_test_y = torch.from_numpy(test_x), test_y.reshape(-1, 1)
model.eval()
A = model(batch_test_x)
Yhat = np.where(A > 0.5, 1, 0)

m = batch_test_x.shape[0]
for i in range(m):
    print('#%3d: predict: %.4f | label: %d' % (i, A[i, 0], batch_test_y[i, 0]))

print('acc %.4f' % ((Yhat == batch_test_y).sum() / m))
