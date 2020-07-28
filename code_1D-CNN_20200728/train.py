import os
import time
import random
import numpy as np
import input_audio_datas
import data_split
import CNN_1D_6group
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

label_name_to_index = {'ceramic tableware': 0,
                       'empty plastic bottle': 1,
                       'empty plastic storage case': 2,
                       'empty tetra pack': 3,
                       'empty tetra pack without the straw': 4,
                       'glass bottle': 5,
                       'glass bottle without the cap': 6,
                       'metal button': 7,
                       'plastic bottle with the filling': 8,
                       'plastic storage case mixed with other waste': 9,
                       'plastic tableware': 10,
                       'resin button': 11}

audio_path = '/home/lg/tensorflow_project/ASR/dataset/data fusion/20200716/Audio/audio_split/train/'
log_dir = '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200717/log/3/'
validation_ratio = 0.1
audio_width = 4410
class_num = 12
batch_size = 792
learning_rate = 1e-3

Input_audios, Input_audio_labels = input_audio_datas.load_datas(
    audio_path, label_name_to_index)

state = np.random.get_state()
np.random.shuffle(Input_audios)
np.random.set_state(state)
np.random.shuffle(Input_audio_labels)

train_audios, train_audio_labels, validation_audios, validation_audio_labels = data_split.train_validation_split(
    Input_audios, Input_audio_labels, validation_ratio)


model = CNN_1D_6group.creation_1dcnn(audio_width, class_num)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_acc', patience=50,  min_delta=0, mode='auto', verbose=1)

checkpoint = ModelCheckpoint(
    log_dir + 'model_{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')


model.fit(train_audios, train_audio_labels, validation_data=(validation_audios, validation_audio_labels),
          epochs=2500, batch_size=batch_size, verbose=1, callbacks=[checkpoint, early_stopping])

print(f'Training is finished!')
