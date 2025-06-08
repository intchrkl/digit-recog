import numpy as np
from layers import Linear, ReLU
from loss import SoftmaxCrossEntropy

class NeuralNet:
    
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr

        self.linear1 = Linear(input_dim, hidden_dim, lr)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_dim, output_dim, lr)
        self.sce = SoftmaxCrossEntropy()

    def forward(self, x, label):
        a = self.linear1.forward(x)
        z = self.relu.forward(a)
        b = self.linear2.forward(z)
        y_hat, loss = self.sce.forward(b, label)
        return y_hat, loss

    def backward(self):
        grad_b = self.sce.backward()
        grad_z = self.linear2.backward(grad_b)
        grad_a = self.relu.backward(grad_z)
        self.linear1.backward(grad_a)

    def step(self):
        self.linear1.step()
        self.linear2.step()

    def predict(self, x):
        a = self.linear1.forward(x)
        z = self.relu.forward(a)
        b = self.linear2.forward(z)
        probs = self.sce.softmax(b)
        prediction = int(np.argmax(probs))
        return prediction, probs

    def train(self, X, y, epochs=5, verbose=True):
        """
        Train the network using SGD.
        :param X: input data of shape (num_samples, input_dim)
        :param y: true labels of shape (num_samples,)
        :param epochs: number of training epochs
        :return: 
            list of training losses per epoch
        """
        losses = []
        n = X.shape[0]
        for epoch in range(epochs):
            idxs = np.random.permutation(n)
            X_shuffled = X[idxs]
            y_shuffled = y[idxs]

            epoch_loss = 0.0

            for i in range(n):
                x_i = X_shuffled[i].reshape(-1, 1)
                y_i = y_shuffled[i]

                _, loss = self.forward(x_i, y_i)
                self.backward()
                self.step()

                epoch_loss += loss

            avg_loss = epoch_loss / n
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return losses


    def evaluate(self, X, y):
        """
        Evaluate the model on input data.
        :param X: input data of shape (num_samples, input_dim)
        :param y: true labels of shape (num_samples,)
        :return: 
            accuracy (float), predictions (list of int)
        """

        n = X.shape[0]
        predictions = []

        correct = 0
        for i in range(n):
            x_i = X[i].reshape(-1, 1)
            pred, _ = self.predict(x_i)
            predictions.append(pred)
            if pred == y[i]:
                correct += 1

        loss = 1 - (correct / n)
        return loss, predictions