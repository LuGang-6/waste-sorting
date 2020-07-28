import tensorflow as tf


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


def inference(datas_batch, label_batch, batch_size, n_class, learning_rate):
    with tf.variable_scope('conv1d_layer1') as scope:
        x_data = tf.reshape(datas_batch, shape=[batch_size, 4410, 1])
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

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label_batch)
        loss = tf.reduce_mean(cross_entropy, name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.nn.in_top_k(logits, label_batch, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, loss, train_step, accuracy
