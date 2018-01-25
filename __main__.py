import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os, cv2, GenData


def categorical(num_classes, classification):
    '''Creates Classification One Hot Values

    :param num_classes: int representing the total number of classes

    :param classification: int representing classification value

    :returns: list representing the one hot classification based off of the given values '''
    return [0]*classification + [1] + [0]*(num_classes-classification-1)


def get_image_shape(filepath):
    '''Returns image dimensions for a given image path

    :param filepath: string representing the filepath for a sample image

    :returns: a list of the image dimensions
        Example: [width, height, channels]'''
    width, height, channels = cv2.imread(filepath).shape
    return [width * height * channels]


filenames = os.listdir(GenData.FUNCTIONS_DIRECTORY_NAME)
filepaths = [os.path.join(GenData.FUNCTIONS_DIRECTORY_NAME, filename) for filename in filenames]

n_classes = len(filenames)
input_shape = [None] + get_image_shape(filepaths[0])

x = tf.placeholder("float", input_shape)
y = tf.placeholder("float", [None, n_classes])

logits = np.array([cv2.imread(filepath) for filepath in filepaths]).reshape([n_classes] + input_shape[1:])
labels = np.array([categorical(n_classes, _) for _ in range(n_classes)])
classification = {str(_): filepath for _, filepath in enumerate(filepaths)}



keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x, n_classes):
    width, height, channels = cv2.imread("funcDirectory/sin.jpeg").shape
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([(width // 3) * (height // 3) * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, width, height, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 86016])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, 0.8)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x, n_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)




    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            print('Optimization Begin')
            _, c = sess.run([optimizer, cost], feed_dict={x: logits, y: labels})
            print('Epoch', epoch, 'Finished, loss: ', c)


train_neural_network(x)