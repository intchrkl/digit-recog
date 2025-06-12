import argparse
import numpy as np
from model import NeuralNet
from utils import Data

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the digit recognizer.")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("-s", "--save", type=str, help="Path to save model weights (as .npz)")
    parser.add_argument("-l", "--load", type=str, help="Path to load model weights (as .npz")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to train with: letters, digits, balanced")
    return parser.parse_args()

def load_data(dataset="digits"):
    if dataset == "letters":
        X_train = Data.parse_images('data/letters/emnist-letters-train-images-idx3-ubyte', is_emnist=True)
        y_train = Data.parse_labels('data/letters/emnist-letters-train-labels-idx1-ubyte', is_emnist=True)
        X_test = Data.parse_images('data/letters/emnist-letters-test-images-idx3-ubyte', is_emnist=True)
        y_test = Data.parse_labels('data/letters/emnist-letters-test-labels-idx1-ubyte', is_emnist=True)
    elif dataset == "balanced":
        X_train = Data.parse_images('data/mnist/train-images.idx3-ubyte')
        y_train = Data.parse_labels('data/mnist/train-labels.idx1-ubyte')
        X_test = Data.parse_images('data/mnist/t10k-images.idx3-ubyte')
        y_test = Data.parse_labels('data/mnist/t10k-labels.idx1-ubyte')
    else:
        X_train = Data.parse_images('data/balanced/emnist-balanced-train-images-idx3-ubyte', is_emnist=True)
        y_train = Data.parse_labels('data/balanced/emnist-balanced-train-labels-idx1-ubyte', is_emnist=True)
        X_test = Data.parse_images('data/balanced/emnist-balanced-test-images-idx3-ubyte', is_emnist=True)
        y_test = Data.parse_labels('data/balanced/emnist-balanced-test-labels-idx1-ubyte', is_emnist=True)
    return X_train, y_train, X_test, y_test

# Arguments
args = parse_args()
X_train, y_train, X_test, y_test = load_data(args.dataset)

input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))
nn = NeuralNet(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, lr=0.01)

if args.load:
    weights = np.load(args.load)
    nn.linear1.W = weights['W1']
    nn.linear1.b = weights['b1']
    nn.linear2.W = weights['W2']
    nn.linear2.b = weights['b2']
    print(f"Loaded weights from {args.load}")
else:
    # Training
    print("Training...")
    losses = nn.train(X_train, y_train, epochs=args.epochs)
    print("Training complete.")

    if args.save:
        np.savez(args.save,
                W1=nn.linear1.W, b1=nn.linear1.b,
                W2=nn.linear2.W, b2=nn.linear2.b)
        print(f"Weights saved to {args.save}")

# Evaluation
input("Begin evaluation?")
print("Evaluating...")
loss, predictions, incorrect = nn.evaluate(X_test, y_test)
print(f"Test error rate: {loss:.4f}")