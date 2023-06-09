import tensorflow as tf
from tensorflow import keras
import numpy.typing as npt

from .model import create_model
from .loader import prepare_data
from . import config

def train(x_train: npt.NDArray, y_train: npt.NDArray, x_val: npt.NDArray, y_val: npt.NDArray, checkpoint_path: str, epochs: int, batch_size: int):

    model = create_model((None, *x_train.shape[2:]))
    # memory allocation
    ml_config = tf.compat.v1.ConfigProto()
    ml_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=ml_config)

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # checkpoint callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)


    # Fit the model to the training data.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, cp_callback],
    )
    
    return model


if __name__ == '__main__':
    # disable GPU
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print('Start training')
    #checkpoint_path = "data/checkpoints/cp-{epoch:04d}.ckpt"
    x_train, y_train, x_val, y_val = prepare_data(config.RC_DATA_PATH, config.GT_DATA_PATH)
    train(x_train, y_train, x_val, y_val, config.CHECKPOINT_DIR, config.EPOCHS, config.BATCH_SIZE)
