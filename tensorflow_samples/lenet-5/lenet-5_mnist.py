# coding=utf8


import tensorflow as tf
import numpy as np
import os
import time
import glob
from skimage import io, transform


parameters = {
    'w': 32,
    'h': 32,
    'c': 1,
    'cf1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], dtype=tf.float32, stddev=0.1, name='filter1')),
    'cf2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], dtype=tf.float32, stddev=0.1, name='filter2')),
    'fc1': tf.Variable(tf.truncated_normal([5*5*16, 120], dtype=tf.float32, stddev=0.1), name='fc1'),
    'fc2': tf.Variable(tf.truncated_normal([120, 84], dtype=tf.float32, stddev=0.1), name='fc2'),
    'fc3': tf.Variable(tf.truncated_normal([84, 10], dtype=tf.float32, stddev=0.1), name='fc2'),
    'cb1': tf.Variable(tf.random_normal([6])),
    'cb2': tf.Variable(tf.random_normal([16])),
    'bc1': tf.Variable(tf.random_normal([120])),
    'bc2': tf.Variable(tf.random_normal([84])),
    'bc3': tf.Variable(tf.random_normal([10])),
    's1': 1,
    's2': 1,
    'k1': 2,
    'k2': 2,
    'ps1': 2,
    'ps2': 2,
    'dropout': 0.5
}

# 数据集地址
path = os.getcwd()
train_path = path + '\\input\\mnist\\train\\'
test_path = path + '\\input\\mnist\\test\\'


# 读取图片及其标签
def read_image(path):
    label_dir = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    images = []
    labels = []
    for index, folder in enumerate(label_dir):
        for img in glob.glob(folder + '/*.png'):
            image = io.imread(img)
            image = transform.resize(image, (parameters['w'], parameters['h'], parameters['c']))
            images.append(image)
            labels.append(index)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


# 定义卷积层
def conv2d(x, f, b, s, padding):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, f, [1, s, s, 1], padding), b))


# 定义池化层
def max_pool(x, k, s,padding):
    return tf.nn.max_pool(x, [1, k, k, 1], [1, s, s, 1], padding)


# 每次获取batch_size样本进行训练或测试
def get_batch(data, label, batch_size):
    for start_index in range(1, len(data) - batch_size, batch_size):
        slice_index = slice(start_index, start_index + batch_size)
        yield data[slice_index], label[slice_index]


def lenet(parameter):
    # 获取图片集数据
    train_data, train_label = read_image(train_path)
    test_data, test_label = read_image(test_path)

    # 打乱训练数据和测试数据
    train_image_num = len(train_data)
    train_image_index = np.arange(train_image_num)
    np.random.shuffle(train_image_index)
    train_data = train_data[train_image_index]
    train_label = train_label[train_image_index]

    test_image_num = len(test_data)
    test_image_index = np.arange(test_image_num)
    np.random.shuffle(test_image_index)
    test_data = test_data[test_image_index]
    test_label = test_label[test_image_index]

    x = tf.placeholder(tf.float32, [None, parameter['w'], parameter['h'], parameter['c']], name='x')
    y = tf.placeholder(tf.int32, [None], name='y')

    # 第一卷积层,过滤器尺寸为5x5，步长为1，数量为6,池化过滤器尺寸为2x2,步长为2， 32x32x1 -> 28x28x6 -> 14x14x6
    conv1 = conv2d(x, parameter['cf1'], parameter['cb1'], parameter['s1'], padding='VALID')
    pool1 = max_pool(conv1, parameter['k1'], parameter['ps1'], padding='SAME')

    # 第二卷积层,过滤器尺寸为5x5,步长为1，数量为16,池化过滤器为2x2,步长为2， 14x14x6 -> 10x10x16 -> 5x5x16
    conv2 = conv2d(pool1, parameter['cf2'], parameter['cb2'], parameter['s2'], padding='VALID')
    pool2 = max_pool(conv2, parameter['k2'], parameter['ps2'], padding='SAME')

    # 第一全连接层,nodes=5x5x16=400, 400 -> 120
    shape = pool2.get_shape()
    reshape = tf.reshape(pool2, [-1, shape[1].value * shape[2].value * shape[3].value])
    fc1 = tf.nn.relu(tf.matmul(reshape, parameter['fc1']) + parameter['bc1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob=parameter['dropout'])

    # 第二全连接层, 120 -> 84
    fc2 = tf.nn.relu(tf.matmul(fc1_drop, parameter['fc2']) + parameter['bc2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob=parameter['dropout'])

    # 输出层, 84 -> 10
    logits = tf.nn.relu(tf.matmul(fc2_drop, parameter['fc3']) + parameter['bc3'])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'))
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

        # 启动图片增强线程队列
        tf.train.start_queue_runners()

        config = tf.ConfigProto(allow_soft_placement=True)
        # 这一行设置 gpu 随使用增长
        config.gpu_options.allow_growth = True

        sess.run(tf.global_variables_initializer())

        train_num = 1000
        batch_size = 128

        for i in range(train_num):
            train_loss, train_acc, batch_num = 0, 0, 0
            for train_data_batch, train_label_batch in get_batch(train_data, train_label, batch_size):
                _, err, acc = sess.run([train_op, loss, accuracy], feed_dict={x:train_data_batch, y:train_label_batch})
                train_loss += err
                train_acc += acc
                batch_num += 1

            print('train loss:', train_loss/batch_num)
            print('train acc:', train_acc/batch_num)

            test_loss, test_acc, batch_num = 0, 0, 0
            for test_data_batch, test_label_batch in get_batch(test_data, test_label, batch_size):
                err, acc = sess.run([loss, accuracy], feed_dict={x:test_data_batch, y:test_label_batch})
                test_acc += acc
                test_loss += err
                batch_num += 1
            print('test loss:', test_loss/batch_num)
            print('test acc:', test_acc/batch_num)


if __name__ == '__main__':
    start = time.time()
    lenet(parameters)
    end = time.time()
    times = end - start
    print('run time is: ', times)