import os
import time
import random
import numpy as np
import tensorflow as tf
import input_datas_cv
import CNN_1D_6group

n_class = 16
batch_size = 720
capacity = 4800
learning_rate = 1e-3
max_step = 100000
best_validation_accuracy = 0
last_improvement = 0
total_iterations = 0
step_patience = 1000
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
train_data_dir = '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200221/5-folder/dataset4/train'
validation_data_dir = '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200221/5-folder/dataset4/validation'
logs_dir = '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200221/log20200221/15-4'
checkpoint_path = os.path.join(
    '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200221/log20200221/15-4/best_validation', 'model.ckpt')

train_data_list, train_label_list = input_datas_cv.get_train_datas(
    train_data_dir)
train_data_input_list, train_label_input_list = input_datas_cv.data_input(
    train_data_list, train_label_list)
train_batch, train_label_batch = input_datas_cv.get_batch(
    train_data_input_list, train_label_input_list, batch_size, capacity)

val_data_list, val_label_list = input_datas_cv.get_validation_datas(
    validation_data_dir)
val_data_input_list, val_label_input_list = input_datas_cv.data_input(
    val_data_list, val_label_list)
val_batch, val_label_batch = input_datas_cv.get_batch(
    val_data_input_list, val_label_input_list, batch_size, capacity)

is_training = tf.placeholder(dtype=tf.bool, shape=())
datas_batch = tf.cond(is_training, lambda: train_batch,
                      lambda: val_batch)
label_batch = tf.cond(is_training, lambda: train_label_batch,
                      lambda: val_label_batch)

logits, loss, train_step, accuracy = CNN_1D_6group.inference(
    datas_batch, label_batch, batch_size, n_class, learning_rate)

merged = tf.summary.merge_all()

sess = tf.Session()
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(logs_dir + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(logs_dir + '/validation')

train_saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

try:
    start_time = time.time()
    for step in range(max_step):
        if coord.should_stop():
            break

        total_iterations += 1

        tra_summary, tra_loss, _, tra_acc = sess.run(
            [merged, loss, train_step, accuracy], feed_dict={is_training: True})
        train_writer.add_summary(tra_summary, step)
        print('step %d, train loss = %.4f, train accuracy = %.2f%%' %
              (step, tra_loss, tra_acc*100))

        val_summary, val_loss, val_acc = sess.run(
            [merged, loss, accuracy], feed_dict={is_training: False})
        validation_writer.add_summary(val_summary, step)
        print('step %d, validation loss = %.4f, validation accuracy = %.2f%%' %
              (step, val_loss, val_acc*100))

        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
            last_improvement = total_iterations
            train_saver.save(sess, checkpoint_path,
                             global_step=total_iterations)

        if total_iterations-last_improvement > step_patience:
            print('No impronement found in a while, stopping training')
            print('step %d, best validation accuracy = %.2f%%' %
                  (last_improvement, best_validation_accuracy*100))
            break

    end_time = time.time()
    time_trian = end_time-start_time
    print('Time usage: ' + str(round(time_trian, 3)))

except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()
