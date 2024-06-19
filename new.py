import numpy as np
import matplotlib.pyplot as plt

# Define weight and bias
Weight = 0.7
bias = 0.3

# Create data points
start = 0
end = 1
step = 0.02
x = np.arange(start, end, step).reshape(-1, 1)
y = Weight * x + bias

# Split data into training and test sets
train_split = int(0.8 * len(x))
X_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Function to plot predictions
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={'size': 14})
    plt.show()

# Initialize weights and bias
weights = np.random.randn(1)
bias = np.random.randn(1)

# Initial predictions before training
initial_preds = weights * x_test + bias
plot_predictions(test_data=x_test, test_labels=y_test, predictions=initial_preds)

# Define learning rate and number of epochs
learning_rate = 0.0001
epochs = 80000

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_preds = weights * X_train + bias

    # Compute loss (L1 Loss)
    loss = np.mean(np.abs(y_preds - y_train))

    # Compute gradients
    grad_weights = np.mean(np.sign(y_preds - y_train) * X_train)
    grad_bias = np.mean(np.sign(y_preds - y_train))

    # Update weights and bias
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias

    # Evaluation on test data
    test_pred = weights * x_test + bias
    test_loss = np.mean(np.abs(test_pred - y_test))

    # Print model state every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Weights: {weights}, Bias: {bias}, Train Loss: {loss}, Test Loss: {test_loss}")

# Final predictions after training
y_preds = weights * x_test + bias
plot_predictions(test_data=x_test, test_labels=y_test, predictions=y_preds)
