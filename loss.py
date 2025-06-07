import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.y = None
        self.y_hat = None

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))
        return exps / np.sum(exps)

    def cross_entropy(self, y, y_hat):
        return -np.log(y_hat[y])

    def forward(self, logits, label):
        pass

    def backward(self):
        pass
