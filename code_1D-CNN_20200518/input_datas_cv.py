import os
import random
import numpy as np
import tensorflow as tf

_RANDOM_SEED = 0

label_name_to_index = {'adhesive tape': 0,
                       'binder clip': 1,
                       'circuit board': 2,
                       'empty glass bottle': 3,
                       'empty packing box': 4,
                       'empty plastic bottle': 5,
                       'empty plastic storage case': 6,
                       'empty tetra pack': 7,
                       'nail clipper': 8,
                       'paper cup': 9,
                       'plastic bottle with the fill': 10,
                       'plastic storage case mixed with other waste': 11,
                       'pop can': 12,
                       'rollerball pen': 13,
                       'tetra pack with the fill': 14,
                       'tweezer': 15}


def get_train_datas(train_data_dir):
    train_data_list = []
    train_label_list = []
    sub_dirs = [x[0] for x in os.walk(train_data_dir)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        dir_name = os.path.basename(sub_dir)
        dir_path = os.path.join(train_data_dir, dir_name)
        for (dirpath, dirname, filenames) in os.walk(dir_path):
            for filename in filenames:
                filename_path = os.sep.join([dirpath, filename])
                train_data_list.append(filename_path)
                train_label_list.append(
                    label_name_to_index[dirpath.split('/')[-1]])
    return train_data_list, train_label_list


def get_validation_datas(validation_data_dir):
    validation_data_list = []
    validation_label_list = []
    sub_dirs = [x[0] for x in os.walk(validation_data_dir)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        dir_name = os.path.basename(sub_dir)
        dir_path = os.path.join(validation_data_dir, dir_name)
        for (dirpath, dirname, filenames) in os.walk(dir_path):
            for filename in filenames:
                filename_path = os.sep.join([dirpath, filename])
                validation_data_list.append(filename_path)
                validation_label_list.append(
                    label_name_to_index[dirpath.split('/')[-1]])
    return validation_data_list, validation_label_list


def data_input(data_list, label_list):
    data_inputs = []
    temp = []
    for data_path in data_list:
        data_input = []
        data_lines = tf.gfile.FastGFile(data_path).readlines()
        for data_line in data_lines:
            data_line = data_line.strip('\n')
            data_line = data_line.split(',')
            data_input.extend(data_line)
        data_input.pop
        data_input = [float(data) for data in data_input]
        data_inputs.append(data_input)
    temp = np.array([data_inputs, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    data_input_list = list(temp[:, 0])
    label_input_list = list(temp[:, 1])
    label_input_list = [int(float(item))
                        for item in label_input_list]
    return data_input_list, label_input_list


def get_batch(data, label, batch_size, capacity):
    data = tf.cast(data, tf.float32)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([data, label])
    data_batch, label_batch = tf.train.batch(input_queue,
                                             batch_size=batch_size,
                                             num_threads=32,
                                             capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    data_batch = tf.cast(data_batch, tf.float32)
    return data_batch, label_batch
