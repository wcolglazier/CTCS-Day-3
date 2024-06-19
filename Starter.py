#pip install matplotlib numpy==1.26.4  
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

# Initial predictions (untrained model)
initial_weights = 0.1
initial_bias = 0.1
y_preds = initial_weights * x_test + initial_bias

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

# Plot initial predictions
plot_predictions(X_train, y_train, x_test, y_test, y_preds)
