import numpy as np
import random


def train_validation_split(input_data, input_data_label, validation_ratio):
    np.random.seed(0)
    validation_set_size = int(len(input_data)*validation_ratio)
    validation_data = input_data[:validation_set_size]
    train_data = input_data[validation_set_size:]
    validation_label = input_data_label[:validation_set_size]
    train_label = input_data_label[validation_set_size:]

    return train_data, train_label, validation_data,  validation_label


''''
if __name__ == '__main__':
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

    image_path = '/home/lg/tensorflow_project/ASR/dataset/data fusion/20200629/image_data/train_set_datagen/'
    audio_data_dir = '/home/lg/tensorflow_project/ASR/dataset/data fusion/20200629/audio_data/train_set'
    log_dir = '/home/lg/tensorflow_project/ASR/dataset/data fusion/log'
    height = 300
    width = 300
    depth = 3
    validation_ratio = 0.25
    Input_images, Input_image_labels = input_image_datas.load_images(
        image_path, label_name_to_index, height, width)

    audio_data_list, audio_label_list = input_audio_datas.audio_list(
        audio_data_dir, label_name_to_index)

    Input_audios, Input_audio_labels = input_audio_datas.load_audio(
        audio_data_list, audio_label_list)

    train_images, validation_images, train_image_labels, validation_image_labels = train_validation_split(
        Input_images, Input_image_labels, validation_ratio)
    # print(train_images)
    # print(validation_images)
    # print(train_image_labels)
    # print(validation_image_labels)
    # print(np.array(train_images).shape)
    # print(np.array(validation_images).shape)
    # print(np.array(train_image_labels).shape)
    # print(np.array(validation_image_labels).shape)

    train_audios, validation_audios, train_audio_labels, validation_audio_labels = train_validation_split(
        Input_audios, Input_audio_labels, validation_ratio)
    # print(train_audios)
    # print(validation_audios)
    # print(train_audio_labels)
    # print(validation_audio_labels)
    # print(np.array(train_audios).shape)
    # print(np.array(validation_audios).shape)
    # print(np.array(train_audio_labels).shape)
    # print(np.array(validation_audio_labels).shape)
'''''
