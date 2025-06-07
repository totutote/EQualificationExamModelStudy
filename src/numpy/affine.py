import numpy as np

class Affine:
    def __init__(self, W, b):
        self.panams = [W, b]

    def forward(self, x):
        W, b = self.panams
        x = np.dot(x, W) + b
        return x

class Relu:
    def forward(self, x):
        x[x < 0] = 0
        return x
        
class Main:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.affine1 = Affine(np.random.randn(input_dim, hidden_dim), np.random.randn(hidden_dim))
        self.affine2 = Affine(np.random.randn(hidden_dim, output_dim), np.random.randn(output_dim))
        self.relu = Relu()

    def predict(self, x):
        x = self.affine1.forward(x)
        x = self.relu.forward(x)
        x = self.affine2.forward(x)
        x = self.relu.forward(x)
        return x

if __name__ == "__main__":
    x = np.random.randn(2, 3)
    print(x)
    test_instance = Main(3, 4, 3)
    result = test_instance.predict(x)
    print(result)