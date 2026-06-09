from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.layers_helper import *

# Parameters file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.append(str(PROJECT_ROOT / "configs"))
yaml_file = PROJECT_ROOT / "configs"/ "real_cnn_params.yaml"

def data_loading():
    data_path = PROJECT_ROOT / "data" / "real_model"

    train_data = np.load(data_path / "train.npz")
    val_data = np.load(data_path / "val.npz")

    X_train = train_data["X"][..., np.newaxis]
    y_train = train_data["y"]

    X_val = val_data["X"][..., np.newaxis]
    y_val = val_data["y"]

    return X_train, y_train, X_val, y_val

def history_saving(history, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(results_dir / "training_history.json", "w") as file:
        json.dump(history_dict, file, indent=2)

def results_saving(history, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    
    X_train, y_train, X_val, y_val = data_loading()
    
    params = load_params(yaml_file)
    
    model_layers = params["cnn_layers"]

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
        elif type == "dense":
            layers.append(dense_layer(**layer_config))
        elif type == "global_max_pooling1d":
            layers.append(global_max_pooling_layer())

    model = keras.Sequential([
        keras.layers.Input(shape=X_train.shape[1:])
    ])
    for layer in layers:
        model.add(layer)

    # Collect parameters for compiling and training
    training_and_compiling = params["training_and_compiling"]

    learning_rate = float(training_and_compiling["learning_rate"])
    weight_decay = float(training_and_compiling["weight_decay"])
    loss = training_and_compiling["loss"]
    metrics = training_and_compiling["metrics"]
    batch_size = training_and_compiling["batch_size"]
    epochs = training_and_compiling["epochs"]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
                       loss=loss, metrics=metrics)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                                  batch_size=batch_size, callbacks=[early_stop])

    history_saving(history, RESULTS_DIR)

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    with open(results_dir / "real_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "real_model.keras")

if __name__ == "__main__":
    main()
