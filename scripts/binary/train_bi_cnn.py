from pathlib import Path
import sys
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
import tensorflow as tf

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.layers_helper import *

SEED = 42
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Parameters file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "binary"
sys.path.append(str(PROJECT_ROOT / "configs"))
yaml_file = PROJECT_ROOT / "configs"/ "bi_cnn_params.yaml"

def data_loading():
    data_path = PROJECT_ROOT / "data" / "binary_model"

    train_data = np.load(data_path / "train.npz")
    val_data = np.load(data_path / "val.npz")

    X_train = train_data["X"][..., np.newaxis]
    y_train = train_data["y"]

    X_val = val_data["X"][..., np.newaxis]
    y_val = val_data["y"]

    return X_train, y_train, X_val, y_val

def prepare_layers(model_layers, input_shape):

    layers = []

    for layer in model_layers:
        layer_config = layer.copy()
        type = layer_config["type"]
        layer_config.pop("type")

        if type == "conv1d":
            layers.append(conv_layer(**layer_config))
        elif type == "max_pooling1d":
            layers.append(max_pooling_layer(**layer_config))
        elif type == "dropout":
            layers.append(dropout_layer(**layer_config))
        elif type == "batch_norm":
            layers.append(batch_norm_layer())
        elif type == "activation":
            layers.append(activations_layer(**layer_config))
        elif type == "dense":
            layers.append(dense_layer(**layer_config))
        elif type == "global_max_pooling1d":
            layers.append(global_max_pooling_layer())
        elif type == "global_avg_pooling1d":
            layers.append(global_avg_pooling_layer())

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape)
    ])
    for layer in layers:
        model.add(layer)

    return model

def run_training(model, training_and_compiling, X_train, y_train, X_val, y_val):
    learning_rate = float(training_and_compiling["learning_rate"])
    weight_decay = float(training_and_compiling["weight_decay"])
    loss = training_and_compiling["loss"]
    metrics = training_and_compiling["metrics"]
    batch_size = training_and_compiling["batch_size"]
    epochs = training_and_compiling["epochs"]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
                  loss=loss, metrics=metrics)

    initial_train_loss, initial_train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    initial_val_loss, initial_val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=batch_size, callbacks=[early_stop], shuffle=True)

    history.history["loss"] = [initial_train_loss] + history.history["loss"]
    history.history["accuracy"] = [initial_train_accuracy] + history.history["accuracy"]
    history.history["val_loss"] = [initial_val_loss] + history.history["val_loss"]
    history.history["val_accuracy"] = [initial_val_accuracy] + history.history["val_accuracy"]

    return history

def history_saving(history, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(results_dir / "bi_training_history.json", "w") as file:
        json.dump(history_dict, file, indent=2)

def results_saving(history, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    with open(results_dir / "bi_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:

    # Load the data
    X_train, y_train, X_val, y_val = data_loading()

    # Load the params from configs/bi_cnn_params.yaml
    params = load_params(yaml_file)

    # Load the order of layers and set up the model
    model_layers = params["cnn_layers"]
    model = prepare_layers(model_layers, X_train.shape[1:])

    # Conduct training
    history = run_training(model, params["training_and_compiling"], X_train, y_train, X_val, y_val)

    # Save the history into a .json file
    history_saving(history, RESULTS_DIR)

    # Save the final metrics into a .json file
    results_saving(history, RESULTS_DIR)

    # Save the model for use in evaluation
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "binary_model.keras")

if __name__ == "__main__":
    main()
