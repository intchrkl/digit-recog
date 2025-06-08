import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.y = None
        self.y_hat = None

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))
        return exps / np.sum(exps)

    def cross_entropy(self, probs, label):
        return -np.log(probs[label])

    def forward(self, logits, label):
        self.y = label
        self.y_hat = self.softmax(logits)
        loss = self.cross_entropy(self.y_hat, label)
        return self.y_hat, loss

    def backward(self):
        grad = self.y_hat.copy()
        grad[self.y] -= 1
        return grad
