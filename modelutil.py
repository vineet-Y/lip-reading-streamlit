# modelutil.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, LSTM, Dense, Dropout, Bidirectional,MaxPool3D, Activation, TimeDistributed, Flatten)

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # Prefer the H5 weights file (works with modern Keras/TF)
    h5_path = os.path.join("models-checkpoint", "lipnet.weights.h5")
    if os.path.exists(h5_path):
        model.load_weights(h5_path)
    else:
        # Fallback for your local old setup if needed
        ckpt_path = os.path.join("models-checkpoint", "checkpoint")
        model.load_weights(ckpt_path)

    return model
