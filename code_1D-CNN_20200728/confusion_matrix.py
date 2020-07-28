import os
import math
import random
import numpy as np
import input_audio_datas_cm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    # plt.title(title)
    # plt.colorbar()
    xlocation = np.array(range(len(labels_index)))
    plt.xticks(xlocation, labels_index)
    plt.yticks(xlocation, labels_index)
    plt.xlabel('Predicted labels', font)
    plt.ylabel('True labels', font)


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


labels_index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 6}
font1 = {'family': 'Times New Roman'}
tick_marks = np.array(range(len(labels_index)))+0.5

audio_path = '/home/lg/tensorflow_project/ASR/dataset/data fusion/20200716/Audio/audio_split/test/'
audio_width = 4410
class_num = 12

Test_audios, Test_audio_labels = input_audio_datas_cm.load_datas(
    audio_path, label_name_to_index)

model = load_model(
    '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200717/log/2/model_164-0.5632-0.7576.hdf5')

y = model.predict(Test_audios)
pred_labels = np.argmax(y, axis=1)

j = 0
for k in range(len(pred_labels)):
    if pred_labels[k] == Test_audio_labels[k]:
        j += 1
acc = j/len(Test_audio_labels)
print(acc)


i = 0
y_pre = []
while i < 360:
    y_pre.append(pred_labels[i])
    i += 1
cm = metrics.confusion_matrix(Test_audio_labels, y_pre)
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
                 fontsize=5, va='center', ha='center', fontdict=font1)
    else:
        plt.text(x_val, y_val, '%0.2f' % (c,), color='black',
                 fontsize=5, va='center', ha='center', fontdict=font1)
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')

# plt.grid(True, which='minor', linestyle='-')
# plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(
    cm_normalized, title='Normalized confusion matrix')
plt.savefig(
    '/home/lg/tensorflow_project/ASR/dataset/1D_CNN/20200717/confusion_matrix.jpg', format='jpg')
plt.show()
