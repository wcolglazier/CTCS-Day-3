import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define weight and bias
Weight = 0.7
bias = 0.3

# Create data points
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
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


# Define Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# Set random seed for reproducibility
torch.manual_seed(42)
model_0 = LinearRegressionModel()



# Make initial predictions with the untrained model
with torch.inference_mode():
    y_preds = model_0(x_test)

# Plot initial predictions
plot_predictions(X_train, y_train, x_test, y_test, y_preds)

# Define loss function and optimizer
# learning rate
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Training loop
epochs = 7480
for epoch in range(epochs):
    model_0.train()  # Set model to training mode

    # Forward pass
    y_preds = model_0(X_train)

    # Compute loss
    loss = loss_fn(y_preds, y_train)

    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Evaluation on test data
    model_0.eval()  # Set model to evaluation mode
    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test)

    # Print model state every 10 epochs
    if epoch % 10 == 0:
        print(model_0.state_dict())

# Final predictions after training
with torch.inference_mode():
    y_preds = model_0(x_test)
    plot_predictions(test_data=x_test, test_labels=y_test, predictions=y_preds.detach())
