from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K


def build_dense_layer(input_layer, units, activation='relu', kernel_regularizer = None, kernel_initializer = None, batch_normalized = True, dropout = True, dropout_rate = 0.8, name=None):

    # add regularlzers
    # https://keras.io/regularizers/
    layer = layers.Dense(units=units, activation=activation, kernel_regularizer = kernel_regularizer,  kernel_initializer = kernel_initializer, name = name)(input_layer)
    if batch_normalized:
        layer = layers.BatchNormalization()(layer)
    if dropout:
        layer = layers.Dropout(dropout_rate)(layer)

    return layer