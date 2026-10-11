
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Iris Neural Network API",
    description="Predict Iris flower species using a trained NumPy neural network.",
    version="1.0.0",
)

# Load the trained model weights
weights = np.load("iris_model_weights.npz")

W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]
W3 = weights["W3"]
b3 = weights["b3"]

# Load preprocessing values and class names
preprocessing = np.load("iris_preprocessing.npz", allow_pickle=True)

scaler_mean = preprocessing["scaler_mean"]
scaler_scale = preprocessing["scaler_scale"]
classes = preprocessing["classes"].astype(str)


# Input data validation
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


def predict_iris(values):
    # Convert input into a NumPy array
    X = np.array([values], dtype=float)

    # Apply the same standardization used during training
    X = (X - scaler_mean) / scaler_scale

    # Neural network forward propagation
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)

    Z2 = A1 @ W2 + b2
    A2 = np.maximum(0, Z2)

    Z3 = A2 @ W3 + b3

    # Softmax output
    exp_values = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    # Select the class with the highest probability
    predicted_index = int(np.argmax(probabilities, axis=1)[0])

    return {
        "predicted_species": str(classes[predicted_index]),
        "confidence": round(
            float(probabilities[0, predicted_index]) * 100, 2
        ),
        "probabilities": {
            str(classes[i]): round(float(probabilities[0, i]) * 100, 2)
            for i in range(len(classes))
        },
    }


@app.get("/")
def home():
    return {
        "message": "Iris Neural Network API is running!",
        "docs": "/docs",
    }


@app.post("/predict")
def predict(data: IrisInput):
    values = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]

    return predict_iris(values)
