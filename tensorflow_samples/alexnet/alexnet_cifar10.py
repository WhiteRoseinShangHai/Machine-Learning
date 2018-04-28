# -*- coding: utf8 -*-
from __future__ import division

import tensorflow as tf
import numpy as np
import cifar10_input
import os
import time


path = os.getcwd()
steps = 10000
batch_size = 128
data_dir = path + '\\input\\'

parameters = {
    'f1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=1e-1), name='f1'),
    'f2': tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=1e-1), name='f2'),
    'f3': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='f3'),
    'f4': tf.Variable(tf.truncated_normal([3, 3, 64, 256], dtype=tf.float32, stddev=1e-1), name='f4'),
    'f5': tf.Variable(tf.truncated_normal([3, 3, 256, 128], dtype=tf.float32, stddev=1e-1), name='f5'),
    'fc1': tf.Variable(tf.truncated_normal([128*24*24, 1024], dtype=tf.float32, stddev=1e-2), name='fc1'),
    'fc2': tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=1e-2), name='fc2'),
    'softmax': tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=1e-2), name='fc3'),
    'bw1': tf.Variable(tf.random_normal([32])),
    'bw2': tf.Variable(tf.random_normal([64])),
    'bw3': tf.Variable(tf.random_normal([64])),
    'bw4': tf.Variable(tf.random_normal([256])),
    'bw5': tf.Variable(tf.random_normal([128])),
    'bc1': tf.Variable(tf.random_normal([1024])),
    'bc2': tf.Variable(tf.random_normal([1024])),
    'bs': tf.Variable(tf.random_normal([10])),
    's1': 1,
    's2': 1,
    's3': 1,
    's4': 1,
    's5': 1,
    'k1': 2,
    'k2': 2,
    'k5': 2,
    'r': 4,
    'alpha': 0.001 / 9.0,
    'beta': 0.75,
    'bias': 1.0,
    'ps1': 2,
    'ps2': 2,
    'ps5': 2
}


def max_pool(x, k, s, name, padding):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],strides=[1, s, s, 1], padding=padding, name=name)


def lrn(x, R, alpha, beta, bias):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha, beta=beta, bias=bias)


def conv2d(x, f, b, s, padding):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, f, strides=[1, s, s, 1], padding=padding), b))


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def alexNet(parameter, dropout):
    """
    五个卷积层，两个全连接层，一个输出层
    """

    #获取要训练的图片数据集
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])

    # 第一卷积层，卷积过滤器尺寸为3x3,步长为1，数量为32，卷积过滤器尺寸为2x2,步长为2
    conv1 = conv2d(image_holder, parameter['f1'], parameter['bw1'], parameter['s1'], padding='SAME')
    lrn1 = lrn(conv1, parameter['r'], parameter['alpha'], parameter['beta'], parameter['bias'])
    pool1 = max_pool(lrn1, parameter['k1'], parameter['ps1'], name='pool1', padding='SAME')

    # 第二卷积层，卷积过滤器尺寸为3x3,步长为1，数量为64，卷积过滤器尺寸为2x2，步长为2
    conv2 = conv2d(pool1, parameter['f2'], parameter['bw2'], parameter['s2'], padding='SAME')
    lrn1 = lrn(conv2, parameter['r'], parameter['alpha'], parameter['beta'], parameter['bias'])
    pool2 = max_pool(lrn1, parameter['k2'], parameter['ps2'], name='pool2', padding='SAME')

    # 第三卷积层，卷积过滤器尺寸为3x3，步长为1，数量为64
    conv3 = conv2d(pool2, parameter['f3'], parameter['bw3'], parameter['s3'], padding='SAME')

    # 第四卷积层，卷积过滤器尺寸为3x3,步长为1，数量为256
    conv4 = conv2d(conv3, parameter['f4'], parameter['bw4'], parameter['s4'], padding='SAME')

    #第五卷积层，卷积过滤器尺寸为3x3，步长为1，数量为128
    conv5 = conv2d(conv4, parameter['f5'], parameter['bw5'], parameter['s5'], padding='SAME')
    pool5 = max_pool(conv5, parameter['k5'], parameter['ps5'], name='pool5', padding='SAME')

    # FC1，128*24*24 -> 1024
    shape = pool5.get_shape()  # 获取第五卷积层输出的结构，并展开
    reshape = tf.reshape(pool5, [-1, shape[1].value * shape[2].value * shape[3].value])
    fc1 = tf.nn.relu(tf.matmul(reshape, parameter['fc1']) + parameter['bc1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob=dropout)

    # FC2, 1024 -> 1024
    fc2 = tf.nn.relu(tf.matmul(fc1_drop, parameter['fc2']) + parameter['bc2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob=dropout)

    # softmax, 1024 -> 10
    logits = tf.add(tf.matmul(fc2_drop, parameter['softmax']), parameter['bs'])
    losses = loss(logits, label_holder)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(losses)

    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

    config = tf.ConfigProto(allow_soft_placement=True)
    # 这一行设置 gpu 随使用增长
    config.gpu_options.allow_growth = True

    # 创建默认session,初始化变量
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 启动图片增强线程队列
    tf.train.start_queue_runners()

    all_vari = tf.global_variables_initializer()
    sess.run(all_vari)

    for step in range(steps):
        batch_xs, batch_ys = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, losses], feed_dict={image_holder: batch_xs, label_holder:batch_ys})

        if step % 20 == 0:
            print('step:%5d. --lost:%6f.' % (step, loss_value))
    print('train over!')

    num_examples = 10000
    import math
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
        true_count += np.sum(predictions)
        step += 1

    precision = true_count / total_sample_count

    print('precision @ 1=%.3f' % precision)


if __name__ == '__main__':
    start = time.time()
    alexNet(parameters, 0.7)
    end = time.time()
    times = end - start
    print(times)