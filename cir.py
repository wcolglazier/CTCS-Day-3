#pip install numpy pandas matplotlib scikit-learn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Data Preparation
n_samples = 1000
X, y = make_circles(n_samples, noise=0.19, random_state=42)

# Convert to DataFrame
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Model Building
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias_vector = np.random.randn(layer_sizes[i + 1])
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            x = self.relu(np.dot(x, self.weights[i]) + self.biases[i])
            self.activations.append(x)
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        self.activations.append(x)
        return x

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_cross_entropy(self, y_true, y_pred):
        y_pred = self.sigmoid(y_pred)
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

    def backward(self, y_true, learning_rate=0.1):
        m = y_true.shape[0]
        y_pred = self.sigmoid(self.activations[-1])
        dL_dz = y_pred - y_true.reshape(-1, 1)
        for i in range(len(self.weights) - 1, -1, -1):
            dz_da = self.activations[i]
            dL_da = np.dot(dL_dz, self.weights[i].T)
            dL_dw = np.dot(dz_da.T, dL_dz) / m
            dL_db = np.sum(dL_dz, axis=0) / m
            self.weights[i] -= learning_rate * dL_dw
            self.biases[i] -= learning_rate * dL_db
            dL_dz = dL_da * (dz_da > 0)


# Define models
model_0 = NeuralNetwork(input_size=2, hidden_layers=[5], output_size=1)
model_1 = NeuralNetwork(input_size=2, hidden_layers=[10, 10], output_size=1)
model_2 = NeuralNetwork(input_size=2, hidden_layers=[10, 10], output_size=1)


# 3. Training Setup
def accuracy_fn(y_true, y_pred):
    y_pred = np.round(y_pred)
    correct = np.sum(y_true == y_pred.squeeze())
    return (correct / len(y_pred)) * 100


# 4. Training Loop
def train_model(model, X_train, y_train, X_test, y_test, epochs=5000, batch_size=32):
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            y_pred_batch = model.forward(X_batch)
            train_loss = model.binary_cross_entropy(y_batch, y_pred_batch)
            model.backward(y_batch)

        train_losses.append(train_loss)

        y_pred_test = model.forward(X_test)
        test_loss = model.binary_cross_entropy(y_test, y_pred_test)
        test_losses.append(test_loss)

        if epoch % 100 == 0:
            train_acc = accuracy_fn(y_train, model.forward(X_train))
            test_acc = accuracy_fn(y_test, y_pred_test)
            print(
                f"Epoch: {epoch} | Loss: {train_loss:.5f}, Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    return train_losses, test_losses


# Train the models
train_losses_0, test_losses_0 = train_model(model_0, X_train, y_train, X_test, y_test)
train_losses_1, test_losses_1 = train_model(model_1, X_train, y_train, X_test, y_test)
train_losses_2, test_losses_2 = train_model(model_2, X_train, y_train, X_test, y_test)


# 5. Define plot_decision_boundary function
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_to_pred = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.sigmoid(model.forward(X_to_pred))
    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary")


# 6. Evaluation and Plotting
plt.figure(figsize=(12, 6))

# Plot decision boundary for training set
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_2, X_train, y_train)

# Plot loss curves
plt.subplot(1, 2, 2)
plt.plot(train_losses_2, label='Train Loss')
plt.plot(test_losses_2, label='Test Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()

plt.show()
