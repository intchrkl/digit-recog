import numpy as np
import struct
import matplotlib.pyplot as plt

class Data:
    def parse_images(path, is_emnist=False):
        with open(path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows * cols)
            if is_emnist:
                # EMNIST data is rotated
                images = np.array([img.reshape(28, 28).T.flatten() for img in images])
            return images / 255.0


    def parse_labels(path, is_emnist=False):
        with open(path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            if is_emnist: labels = labels - 1
            return labels

    def parse_image_from_txt(path):
        """
        Parse a single image stored as space-separated pixel values (optionally with label).
        Returns a (784, 1) column vector normalized to [0, 1] for prediction.
        """
        with open(path, 'r') as f:
            line = f.readline().strip()
            values = list(map(float, line.split()))

            if len(values) != 784:
                raise ValueError(f"Expected 784 pixel values, got {len(values)}.")

            image_array = np.array(values, dtype=np.float32).reshape(-1, 1)
            return image_array

    def preview_data(X, y, select=None):
        rows, cols, = 10, 10
        indices = None
        if select is not None:
            indices = np.where(y == select)[0]
        else:
            indices = np.random.choice(np.array(range(len(y))), rows * cols, False)
        
        indices = indices[:rows * cols]
        plt.subplots(rows, cols)
        for plot_idx, data_idx in enumerate(indices):
            plt.subplot(rows, cols, plot_idx + 1)
            plt.imshow(X[data_idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.show()


    def visualize(image_array, title=None):
        print(image_array)
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        if title: plt.title(title)
        plt.axis('off')
        plt.show()


    def visualize_predict(image_array, prediction, true_label):
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {prediction} | True: {true_label}")
        plt.axis('off')
        plt.show()


    def visualize_from_txt(path):
        """
        Read a txt file containing 784 pixel values (optionally with a label)
        and visualize it as a 28x28 grayscale image.
        """
        with open(path, 'r') as f:
            line = f.readline().strip()
            values = list(map(float, line.split()))

            pixels = values

            image_array = np.array(pixels).reshape(28, 28)
            Data.visualize(image_array)


    def save_as_txt(image, out_path, label=None):
        """
        Save a single image (1D or 2D NumPy array) as a txt file.
        If label is provided, it will be prepended to the line.
        """
        with open(out_path, 'w') as f:
            flat_img = image.flatten()
            pixels = ' '.join(str(p) for p in flat_img)
            if label is not None:
                f.write(f"{label} {pixels}\n")
            else:
                f.write(f"{pixels}\n")

# X_test = Data.parse_images('data/digits/t10k-images.idx3-ubyte')
# y_test = Data.parse_labels('data/digits/t10k-labels.idx1-ubyte')
# X_test = Data.parse_images('data/letters/emnist-letters-test-images-idx3-ubyte', is_emnist=True)
# y_test = Data.parse_labels('data/letters/emnist-letters-test-labels-idx1-ubyte')
# X_test = Data.parse_images('data/letters/emnist-letters-train-images-idx3-ubyte', is_emnist=True)
# y_test = Data.parse_labels('data/letters/emnist-letters-train-labels-idx1-ubyte', is_emnist=True)
# Data.preview_data(X_test, y_test, 5)