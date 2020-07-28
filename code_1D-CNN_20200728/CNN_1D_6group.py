import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling2D, GlobalAveragePooling1D


def creation_1dcnn(audio_width, class_num):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=9, strides=1,
                     padding='same', activation='relu', input_shape=(audio_width, 1)))

    model.add(MaxPooling1D(pool_size=9, strides=2, padding='same'))

    model.add(Conv1D(filters=16, kernel_size=9, strides=1,
                     padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=9, strides=2, padding='same'))

    model.add(Conv1D(filters=32, kernel_size=9, strides=1,
                     padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=9, strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=9, strides=1,
                     padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=9, strides=2, padding='same'))

    model.add(Conv1D(filters=128, kernel_size=9, strides=1,
                     padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=9, strides=2, padding='same'))

    model.add(Conv1D(filters=256, kernel_size=9, strides=1,
                     padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=9, strides=2, padding='same'))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(1000, activation='relu'))

    model.add(Dense(class_num, activation='softmax'))

    return model
