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
    return [width, height, channels]


def model(x, mode):
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2
    )
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
    )
    pool_flat = tf.layers.flatten(
        inputs=pool2
    )
    dense = tf.layers.dense(
        inputs=pool_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode
    )
    out = tf.layers.dense(
        inputs=dropout,
        units=13,
    )
    return out


def train_model(x, y, mode, logits, labels, iterations):
    prediction = model(x, mode)
    probabilities = tf.nn.softmax(prediction)

    with tf.Session() as sess:
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=y,
            logits=prediction
        )

        optimizer = tf.train.AdamOptimizer().minimize(loss)
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            loss_val, opt_func = sess.run([loss, optimizer],
                                          feed_dict={x:logits,
                                                     y:labels,
                                                     mode:True})
            print("Loss {1}: {0}".format(loss_val, i))

        return  (str(np.argmax(sess.run([prediction], feed_dict={
            x:[logits[4]],
            mode:False})[0])), sess.run([probabilities], feed_dict={x:[logits[4]], mode:False}))



def main():
    filenames = os.listdir(GenData.FUNCTIONS_DIRECTORY_NAME)
    filepaths = [os.path.join(GenData.FUNCTIONS_DIRECTORY_NAME, filename) for filename in filenames]

    n_classes = len(filenames)
    input_shape = [None] + get_image_shape(filepaths[0])

    x = tf.placeholder("float", input_shape)
    y = tf.placeholder("float", [None, n_classes])
    mode = tf.placeholder(tf.bool, shape=())

    logits = np.array([cv2.imread(filepath) for filepath in filepaths]).reshape([13]+input_shape[1:])
    labels = np.array([categorical(n_classes, p) for p in range(n_classes)])
    classification = {str(p): filepath for p, filepath in enumerate(filepaths)}
    print(input_shape)
    print(logits.shape)

    keep_rate = 0.8
    keep_prob = tf.placeholder(tf.float32)
    result = train_model(x, y, mode, logits, labels, 40)
    print(classification[result[0]])
    print(result[1])


if __name__ == "__main__":
    main()