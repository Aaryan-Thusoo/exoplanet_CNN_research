import tensorflow as tf
from tensorflow import keras

import pandas as pd

from LightCurve import test_train_lc

from functions.model_analysis_functions import *
import matplotlib.pyplot as plt

def get_misclassified_indices(y_true, y_pred):
    """
    Return the indices where y_true and y_pred do not match.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    np.ndarray
        Array of misclassified indices.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match. Got {y_true.shape} and {y_pred.shape}")

    misclassified_idx = np.where(y_true != y_pred)[0]
    return misclassified_idx

def get_misclassified_info(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match. Got {y_true.shape} and {y_pred.shape}")

    misclassified_idx = np.where(y_true != y_pred)[0]
    return misclassified_idx, y_true[misclassified_idx], y_pred[misclassified_idx]

def plot_an_lc(X, i):
    plt.figure(figsize=(12,4))
    plt.scatter(np.arange(0, len(X[i])), X[i])
    plt.xlabel("Days")
    plt.ylabel("Normalized Flux")
    plt.title(f"Index {i}")

def plot_training_history(history):
    history_dict = history.history

    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    accuracy = history_dict.get("accuracy", [])
    val_accuracy = history_dict.get("val_accuracy", [])

    epochs = range(1, len(loss) + 1)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Training Loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, label="Training Accuracy")
    if val_accuracy:
        plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()

def results(history, y_harder_test, y_pred):
    plot_training_history(history)
    plot_confusion_matrix_2D(y_harder_test, y_pred)