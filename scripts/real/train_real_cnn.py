# region Imports
from pathlib import Path
import sys

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.layers_helper import *

# Parameters file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "configs"))
yaml_file = PROJECT_ROOT / "configs"/ "real_cnn_params.yaml"


def main() -> None:
    params = load_params(yaml_file)

    model_layers = params["cnn_layers"]

    real_layers = []

    for layer in model_layers:
        layer_config = layer.copy()
        type = layer_config["type"]
        layer_config.pop("type")

        if type == "conv1d":
            real_layers.append(conv_layer(**layer_config))
        elif type == "max_pooling1d":
            real_layers.append(max_pooling_layer(**layer_config))
        elif type == "dropout":
            real_layers.append(dropout_layer(**layer_config))
        elif type == "dense":
            real_layers.append(dense_layer(**layer_config))
        elif type == "global_max_pooling1d":
            real_layers.append(global_max_pooling_layer())

    real_model = keras.Sequential(keras.layers.Input(shape=X_real_train.shape[1:]))
    for layer in real_layers:
        real_model.add(layer)

    real_model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4, weight_decay=1e-4),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    real_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    real_history = real_model.fit(X_real_train, y_real_train, validation_data=(X_real_val, y_real_val), epochs=5,
                                  batch_size=32, callbacks=[real_early_stop])

    real_model.save("models/real_model.keras")

if __name__ == "__main__":
    main()
