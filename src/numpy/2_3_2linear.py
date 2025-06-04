import numpy as np

np.random.seed(2)  # 再現性のためのシード設定

# 活性化関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# 損失関数
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# 全結合層
def linear(x, w, b):
    return np.dot(x, w) + b

if __name__ == "__main__":
    In1 = np.random.randn(10, 2)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(3)
    In2 = linear(In1, W1, b1)
    In2 = relu(In2)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(2)
    In2 = linear(In2, W2, b2)
    In2 = relu(In2)
    In2 = softmax(In2)
    print(In2)
