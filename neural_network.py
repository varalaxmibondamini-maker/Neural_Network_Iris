
import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss


# 1. MLflow configuration
mlflow.set_experiment("Iris_Neural_Network")


# 2. Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

X = data[
    ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
].values

y = data["Species"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

number_of_classes = len(encoder.classes_)
print("Classes:", encoder.classes_)


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10,
    stratify=y
)


# 4. Scale using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. One-hot encoding
def one_hot(y, number_of_classes):
    result = np.zeros((len(y), number_of_classes))
    result[np.arange(len(y)), y] = 1
    return result


y_train_encoded = one_hot(y_train, number_of_classes)


# 6. Activation functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 7. Hyperparameters
learning_rate = 0.1
epochs = 65
input_neurons = 4
hidden_layer_1 = 32
hidden_layer_2 = 16
output_neurons = number_of_classes


# 8. Train and track the experiment
with mlflow.start_run(run_name="Iris_NN_65_Epochs"):

    mlflow.log_params({
        "learning_rate": learning_rate,
        "epochs": epochs,
        "input_neurons": input_neurons,
        "hidden_layer_1": hidden_layer_1,
        "hidden_layer_2": hidden_layer_2,
        "output_neurons": output_neurons,
        "test_size": 0.2,
        "random_state": 10,
        "optimizer": "Manual Gradient Descent",
        "hidden_activation": "ReLU",
        "output_activation": "Softmax",
        "loss_function": "Categorical Cross-Entropy"
    })

    np.random.seed(42)

    W1 = np.random.randn(4, 32) * np.sqrt(2 / 4)
    b1 = np.zeros((1, 32))

    W2 = np.random.randn(32, 16) * np.sqrt(2 / 32)
    b2 = np.zeros((1, 16))

    W3 = np.random.randn(16, output_neurons) * np.sqrt(2 / 16)
    b3 = np.zeros((1, output_neurons))

    for epoch in range(epochs):

        # Forward propagation
        Z1 = X_train @ W1 + b1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + b2
        A2 = relu(Z2)

        Z3 = A2 @ W3 + b3
        A3 = softmax(Z3)

        # Categorical cross-entropy loss
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

        # Log loss every epoch
        mlflow.log_metric("training_loss", float(loss), step=epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # 9. Test evaluation
    Z1 = X_test @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    predictions = np.argmax(A3, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    test_loss = log_loss(y_test, A3, labels=np.arange(number_of_classes))

    mlflow.log_metric("test_accuracy", float(accuracy))
    mlflow.log_metric("test_loss", float(test_loss))

    print("\n============================")
    print("NEURAL NETWORK RESULTS")
    print("============================")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nActual Classes:")
    print(y_test)

    print("\nPredicted Classes:")
    print(predictions)

    print("\nPredicted Flower Names:")
    for prediction in predictions:
        print(encoder.inverse_transform([prediction])[0])

    # Save learned parameters and preprocessing objects
    np.savez(
        "iris_model_weights.npz",
        W1=W1, b1=b1,
        W2=W2, b2=b2,
        W3=W3, b3=b3
    )

    np.savez(
        "iris_preprocessing.npz",
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=encoder.classes_
    )

    mlflow.log_artifact("iris_model_weights.npz")
    mlflow.log_artifact("iris_preprocessing.npz")

    mlflow.set_tag("model_type", "NumPy Neural Network")
    mlflow.set_tag("dataset", "Iris")

    print("\nMLflow run completed successfully!")
