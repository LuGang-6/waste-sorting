import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def load_datas(audio_path, label_name_to_index):
    audios = []
    audio_labels = []
    audio_class_name_list = os.listdir(audio_path)
    audio_class_num = len(audio_class_name_list)
    for i in range(0, audio_class_num):
        audio_list = os.listdir(audio_path + audio_class_name_list[i])
        audio_list.sort(key=lambda x: int(x.split('.')[0]))

        for audio in audio_list:
            audio_data = []
            per_audio_path = audio_path + \
                audio_class_name_list[i] + '/' + audio
            data_lines = tf.gfile.FastGFile(per_audio_path).readlines()
            for data_line in data_lines:
                data_line = data_line.strip('\n')
                data_line = data_line.split(',')
                audio_data.extend(data_line)
            audio_data.pop
            audio_data = [float(data) for data in audio_data]
            audios.append(audio_data)
            audio_label_index = label_name_to_index[audio_class_name_list[i]]
            audio_labels.append(audio_label_index)

    input_audios = np.array(audios)
    input_audios = np.expand_dims(input_audios, axis=-1)
    #input_audio_labels = to_categorical(audio_labels, 12)
    #input_audio_labels = np.array(input_audio_labels)
    input_audio_labels = np.array(audio_labels)
    #input_audio_labels = np.expand_dims(input_audio_labels, -1)

    return input_audios, input_audio_labels
