import numpy as np

class Affine:
    def __init__(self, W, b):
        self.panams = [W, b]
        self.W_grads = np.zeros_like(W)
        self.b_grads = np.zeros_like(b)
        self.x = None

    def forward(self, x):
        self.x = x
        W, b = self.panams
        x = np.dot(x, W) + b
        return x

    def backward(self, prev):
        W, b = self.panams
        self.W_grads = np.dot(self.x.T, prev)
        self.b_grads = np.sum(prev, axis=0)
        return np.dot(prev, W.T)
        
class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        x[x < 0] = 0
        return x
    
    def backward(self, prev):
        dx = np.where(self.x > 0, prev, 0)
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, prev):
        return prev * (self.out * (1 - self.out))

class Main:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.affine1 = Affine(np.random.randn(input_dim, hidden_dim), np.random.randn(hidden_dim))
        self.relu1 = Relu()
        self.affine2 = Affine(np.random.randn(hidden_dim, output_dim), np.random.randn(output_dim))
        self.sigmoid = Sigmoid()

    def predict(self, x):
        x = self.affine1.forward(x)
        x = self.relu1.forward(x)
        x = self.affine2.forward(x)
        x = self.sigmoid.forward(x)
        return x
    
    def backward(self, prev):
        prev = self.sigmoid.backward(prev)
        prev = self.affine2.backward(prev)
        prev = self.relu1.backward(prev)
        prev = self.affine1.backward(prev)
        return prev

if __name__ == "__main__":
    x = np.random.randn(2, 3)
    print(x)
    test_instance = Main(3, 4, 3)
    result = test_instance.predict(x)
    print(result)
    result = test_instance.backward(result)
    print(result)
