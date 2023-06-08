from tensorflow import keras

def create_model(input_shape = (None, 64, 64, 1)):
    # Construct the input layer with no definite frame size.
    inp = keras.layers.Input(shape=input_shape) # batch_size, width, height, channels

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)

    # to compress data
    x = keras.layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)

    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
    )

    # model.compile(
    #     loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(),
    # )

    # TODO: keras.losses.binary_crossentropy to MAE

    return model