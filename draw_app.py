import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from model import NeuralNet
from utils import Data

# Load model and weights
input_dim = 784
model = NeuralNet(input_dim=input_dim, hidden_dim=128, output_dim=10)
weights = np.load("params/ten_epochs.npz")
model.linear1.W = weights["W1"]
model.linear1.b = weights["b1"]
model.linear2.W = weights["W2"]
model.linear2.b = weights["b2"]

class DigitDrawApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit")
        self.canvas_size = 280
        self.pixel_size = 10
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.save_button = tk.Button(self, text="Save as TXT", command=self.save_as_grayscale_txt)
        self.save_button.pack()

        self.predict_button = tk.Button(self, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.prediction_label = tk.Label(self, text="", font=("Helvetica", 18))
        self.prediction_label.pack()

    def paint(self, event):
        x1, y1 = event.x - self.pixel_size, event.y - self.pixel_size
        x2, y2 = event.x + self.pixel_size, event.y + self.pixel_size
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.prediction_label.config(text="")

    def save_as_grayscale_txt(self):
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        data = np.asarray(img_resized, dtype=np.float32)
        data = 1.0 - data / 255.0
        flattened = data.flatten()

        with open("draw_digit.txt", "w") as f:
            line = " ".join(f"{value:.8f}" for value in flattened)
            f.write(line + "\n")

        print("Saved as draw_digit.txt")

    def predict_digit(self):
        # Prepare image for prediction
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        data = np.asarray(img_resized, dtype=np.float32)
        data = 1.0 - data / 255.0  # Normalize
        input_vector = data.flatten().reshape(-1, 1)

        # Predict
        prediction, _ = model.predict(input_vector)
        self.prediction_label.config(text=f"Prediction: {prediction}")

        # Uncomment to visualize model's view of the drawing
        # Data.visualize(data, f"Processed Image | Prediction: {prediction}")

if __name__ == "__main__":
    app = DigitDrawApp()
    app.mainloop()
