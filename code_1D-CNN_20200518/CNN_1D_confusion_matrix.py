import os
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics


def get_test_datas(test_data_dir):
    test_data_list = []
    test_label_list = []
    sub_dirs = [x[0] for x in os.walk(test_data_dir)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        dir_name = os.path.basename(sub_dir)
        dir_path = os.path.join(test_data_dir, dir_name)
        for (dirpath, dirname, filenames) in os.walk(dir_path):
            for filename in filenames:
                filename_path = os.sep.join([dirpath, filename])
                test_data_list.append(filename_path)
                test_label_list.append(
                    label_name_to_index[dirpath.split('/')[-1]])
    return test_data_list, test_label_list


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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(X, W):
    return tf.nn.conv1d(X, W, stride=1, padding='SAME')


def max_pool_1D(X):
    return tf.nn.pool(X, window_shape=[9], pooling_type='MAX', padding='SAME', strides=[2])


def inference(datas, batch_size, n_class):
    with tf.variable_scope('conv1d_layer1') as scope:
        x_data = tf.reshape(datas, shape=[batch_size, 4410, 1])
        W_conv1 = weight_variable([9, 1, 8])
        b_conv1 = bias_variable([8])
        L_conv1 = tf.nn.relu(conv1d(x_data, W_conv1)+b_conv1)

    with tf.variable_scope('max_pool_layer2') as scope:
        L_pool1 = max_pool_1D(L_conv1)

    with tf.variable_scope('conv1d_layer3') as scope:
        W_conv2 = weight_variable([9, 8, 16])
        b_conv2 = bias_variable([16])
        L_conv2 = tf.nn.relu(conv1d(L_pool1, W_conv2)+b_conv2)

    with tf.variable_scope('max_pool_layer4') as scope:
        L_pool2 = max_pool_1D(L_conv2)

    with tf.variable_scope('conv1d_layer5') as scope:
        W_conv3 = weight_variable([9, 16, 32])
        b_conv3 = bias_variable([32])
        L_conv3 = tf.nn.relu(conv1d(L_pool2, W_conv3)+b_conv3)

    with tf.variable_scope('max_pool_layer6') as scope:
        L_pool3 = max_pool_1D(L_conv3)

    with tf.variable_scope('conv1d_layer7') as scope:
        W_conv4 = weight_variable([9, 32, 64])
        b_conv4 = bias_variable([64])
        L_conv4 = tf.nn.relu(conv1d(L_pool3, W_conv4)+b_conv4)

    with tf.variable_scope('max_pool_layer8') as scope:
        L_pool4 = max_pool_1D(L_conv4)

    with tf.variable_scope('conv1d_layer9') as scope:
        W_conv5 = weight_variable([9, 64, 128])
        b_conv5 = bias_variable([128])
        L_conv5 = tf.nn.relu(conv1d(L_pool4, W_conv5)+b_conv5)

    with tf.variable_scope('max_pool_layer10') as scope:
        L_pool5 = max_pool_1D(L_conv5)

    with tf.variable_scope('conv1d_layer11') as scope:
        W_conv6 = weight_variable([9, 128, 256])
        b_conv6 = bias_variable([256])
        L_conv6 = tf.nn.relu(conv1d(L_pool5, W_conv6)+b_conv6)

    with tf.variable_scope('max_pool_layer12') as scope:
        L_pool6 = max_pool_1D(L_conv6)

    with tf.variable_scope('fc_layer9') as scope:
        reshape = tf.reshape(L_pool6, shape=[batch_size, -1])
        neurons = reshape.get_shape()[1].value
        W_fc1 = weight_variable([neurons, 1024])
        b_fc1 = bias_variable([1024])
        L_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1)+b_fc1)

    with tf.variable_scope('logits') as scope:
        W_fc2 = weight_variable([1024, n_class])
        b_fc2 = bias_variable([n_class])
        logits = tf.matmul(L_fc1, W_fc2)+b_fc2
    return logits


def losses(logits, label_batch):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label_batch)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar('loss', loss)
    return loss


def training(loss, learning_rate):
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(loss)
    return train_step


def evaluation(logits, label_batch):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.nn.in_top_k(logits, label_batch, 1)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    # plt.title(title)
    # plt.colorbar()
    xlocation = np.array(range(len(labels_index)))
    plt.xticks(xlocation, labels_index)
    plt.yticks(xlocation, labels_index)
    plt.xlabel('(a) Predicted labels', font)
    plt.ylabel('True labels', font)


if __name__ == "__main__":
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
    labels_index = ['0', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9', '10', '11', '12', '13', '14', '15']
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 6}
    font1 = {'family': 'Times New Roman'}
    tick_marks = np.array(range(len(labels_index)))+0.5
    n_class = 16
    test_batch_size = 960
    learning_rate = 1e-3
    test_data_dir = '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200221/test_set'
    logs_train_dir = '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200221/log20200221/3/best_validation'
    test_data_list, test_label_list = get_test_datas(test_data_dir)
    test_data_input_list, test_label_input_list = data_input(
        test_data_list, test_label_list)
    logits = inference(test_data_input_list, test_batch_size, n_class)
    correct_pre = tf.argmax(logits, 1)
    saver = tf.train.Saver(tf.global_variables())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split(
            '/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')

    pred_labels = sess.run([correct_pre])
    i = 0
    y_pre = []
    while i < 960:
        y_pre.append(pred_labels[0][i])
        i += 1
    cm = metrics.confusion_matrix(test_label_input_list, y_pre)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)
    plt.figure(figsize=(18, 12), dpi=350)
    ax = plt.subplot(111)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.tick_params(labelsize=6)

    ind_array = np.array(range(len(labels_index)))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if (c > 0.5):
            plt.text(x_val, y_val, '%0.2f' % (c,), color='white',
                     fontsize=4, va='center', ha='center', fontdict=font1)
        else:
            plt.text(x_val, y_val, '%0.2f' % (c,), color='black',
                     fontsize=4, va='center', ha='center', fontdict=font1)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')

    # plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(
        cm_normalized, title='Normalized confusion matrix')
    plt.savefig(
        '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200221/confusion_matrix_cnn.tif', format='tif')
    plt.show()
    sess.close()
