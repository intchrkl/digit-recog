import numpy as np

class Linear:
    def __init__(self, in_features, out_features, lr=0.01):
        self.lr = lr
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features) # He Initialization
        self.b = np.zeros((out_features, 1))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x # (in_features, 1)
        return self.W @ x + self.b # (out_features, 1)

    def backward(self, grad_output):
        self.dW = grad_output @ self.x.T
        self.db = grad_output
        return self.W.T @ grad_output

    def step(self):
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * self.mask