import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# 1. Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# 2. Separate input and output
X = data[
    ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
].values

y = data["Species"].values


# 3. Convert species names to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Classes:", encoder.classes_)


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10,
    stratify=y
)

# 5. Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6. One-hot encoding
def one_hot(y, number_of_classes):
    result = np.zeros((len(y), number_of_classes))

    for i in range(len(y)):
        result[i, y[i]] = 1

    return result


y_train_encoded = one_hot(y_train, 3)


# 7. Activation functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 8. Initialize weights
np.random.seed(42)

W1 = np.random.randn(4, 32) * np.sqrt(2 / 4)
b1 = np.zeros((1, 32))

W2 = np.random.randn(32, 16) * np.sqrt(2 / 32)
b2 = np.zeros((1, 16))

W3 = np.random.randn(16, 3) * np.sqrt(2 / 16)
b3 = np.zeros((1, 3))


# 9. Learning rate
learning_rate = 0.1


# 10. Training
epochs = 65

for epoch in range(epochs):

    # Forward propagation
    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)


    # Cross entropy loss
    loss = -np.mean(
        np.sum(
            y_train_encoded * np.log(A3 + 1e-8),
            axis=1
        )
    )


    # Backpropagation
    dZ3 = A3 - y_train_encoded

    dW3 = (A2.T @ dZ3) / len(X_train)
    db3 = np.mean(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = (A1.T @ dZ2) / len(X_train)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = (X_train.T @ dZ1) / len(X_train)
    db1 = np.mean(dZ1, axis=0, keepdims=True)


    # Update weights
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


    # Display loss
    if (epoch + 1) % 200 == 0:
        print(
            "Epoch:",
            epoch + 1,
            "Loss:",
            round(loss, 4)
        )


# 11. Testing

Z1 = X_test @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = relu(Z2)

Z3 = A2 @ W3 + b3
A3 = softmax(Z3)


# 12. Predictions
predictions = np.argmax(A3, axis=1)


# 13. Accuracy
accuracy = np.mean(predictions == y_test)

print("\n============================")
print("NEURAL NETWORK RESULTS")
print("============================")

print("Test Accuracy:", round(accuracy * 100, 2), "%")


# 14. Actual vs predicted
print("\nActual Classes:")
print(y_test)

print("\nPredicted Classes:")
print(predictions)


# 15. Flower names
print("\nPredicted Flower Names:")

for prediction in predictions:
    name = encoder.inverse_transform([prediction])[0]
    print(name)