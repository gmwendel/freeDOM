import tensorflow as tf

from freedom.neural_nets.transformations import superstringnet_trafo

def get_superstringnet(labels):

    superstring_input = tf.keras.Input(shape=(86,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = superstringnet_trafo(labels=labels)

    h = t(superstring_input, params_input)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(512, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(1024, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(512, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    superstringnet = tf.keras.Model(inputs=[superstring_input, params_input], outputs=outputs)

    return superstringnet