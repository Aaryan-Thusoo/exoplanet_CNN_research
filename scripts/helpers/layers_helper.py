from tensorflow import keras

def conv_layer(channels, kernel_size, padding, activation=None):
    return keras.layers.Conv1D(channels, kernel_size=kernel_size, padding=padding, activation=activation)

def max_pooling_layer(pool_size):
    return keras.layers.MaxPooling1D(pool_size=pool_size)

def dropout_layer(rate):
    return keras.layers.Dropout(rate)

def dense_layer(units, activation):
    return keras.layers.Dense(units=units, activation=activation)

def global_max_pooling_layer():
    return keras.layers.GlobalMaxPooling1D()

def global_avg_pooling_layer():
    return keras.layers.GlobalAveragePooling1D()

def batch_norm_layer():
    return keras.layers.BatchNormalization()

def activations_layer(activation):
    return keras.layers.Activation(activation)