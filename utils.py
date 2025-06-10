import numpy as np
import struct
import matplotlib.pyplot as plt

class Data:
    def parse_images(path):
        with open(path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows * cols)
            return images / 255.0


    def parse_labels(path):
        with open(path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels


    def visualize(image_array):
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

    def visualize_predict(image_array, prediction, true_label):
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {prediction} | True: {true_label}")
        plt.axis('off')
        plt.show()

    
# print(f"X_train: {X_train.shape}")
# print(f"y_train: {y_train.shape}")
# print(f"X_test: {X_test.shape}")
# print(f"y_test: {y_test.shape}")

# print("Label:", y_train[0])
# print(X_train[0])

# data = Data
# X_train = data.parse_images('data/train-images.idx3-ubyte')
# y_train = data.parse_labels('data/train-labels.idx1-ubyte')
# X_test = data.parse_images('data/t10k-images.idx3-ubyte')
# y_test = data.parse_labels('data/t10k-labels.idx1-ubyte')
# data.visualize(X_train[0])
