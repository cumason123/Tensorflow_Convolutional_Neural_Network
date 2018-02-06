import tensorflow as tf
import numpy as np
import os, cv2
import time
import GenData
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CryptoNeuralNetwork(object):
    def __init__(self):
        self.directories = os.listdir(GenData.FUNCTIONS_DIRECTORY_NAME)
        self.fullpaths = [os.path.join(GenData.FUNCTIONS_DIRECTORY_NAME, item) for item
                                 in self.directories]

        self.image_shape = GenData.get_image_shape(
            os.path.join(
                self.fullpaths[0],
                self.directories[0]+"1.jpeg"
            )
        )

        self.n_classes = len(self.directories)

        self.x = tf.placeholder("float", [None] + self.image_shape)
        self.y = tf.placeholder("float", [None, self.n_classes])
        self.train = tf.placeholder(tf.bool, shape=())
        self.learn_rate = tf.placeholder("float", shape=())

        self.logits=[]
        self.labels=[]

        self.out = []

        GenData.create_directory("memory/")
        self.memory_path = "memory/model.ckpt"

    def load_data(self):
        """Loads train data from func_directory
        """
        for directory in self.fullpaths:
            directory_list = os.listdir(directory)
            for filename in directory_list:
                self.logits.append(cv2.imread(os.path.join(directory, filename)))

        for category, directory in enumerate(self.fullpaths):
            directory_list = os.listdir(directory)
            for filename in directory_list:
                self.labels.append((GenData.categorical(self.n_classes, category)))

        self.logits = np.array(self.logits)
        self.labels = np.array(self.labels)

        print(self.logits.shape)


    def model(self, x):
        """Convolutional Neural Network Model

        :params x: tensorflow placeholder variable for logits

        :params train: bool value indicating whether to use the dropout layer

        :returns: tensorflow convolutional neural model
        """
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="conv1"
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2,
            name="pool1"
        )
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="conv2"
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2,
            name="pool2"
        )
        pool_flat = tf.layers.flatten(
            inputs=pool2,
            name="pool_flat"
        )
        dense = tf.layers.dense(
            inputs=pool_flat,
            units=1024,
            activation=tf.nn.relu,
            name="dense"
        )
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.6,
            training=self.train,
            name="dropout"
        )
        out = tf.layers.dense(
            inputs=dropout,
            units=self.n_classes,
            name="out"
        )
        return out

    @staticmethod
    def train_model(iterations=20, save_checkpoint=30, load_checkpoint=False):
        with CryptoNeuralNetwork() as self:
            if self.logits == []:
                self.load_data()
            prediction = self.model(self.x)
            probability = tf.nn.softmax(prediction)

            cost = tf.losses.softmax_cross_entropy(
                onehot_labels=self.y,
                logits=prediction,
            )
            optimizer = tf.train.AdamOptimizer().minimize(cost)
            with tf.Session() as sess:
                if load_checkpoint:
                    if os.path.exists("memory/"):
                        tf.train.Saver().restore(sess, self.memory_path)
                        print("Sess Restored")
                    else:
                        print("Checkpoint: {0} could not be found. Continuing".format(
                            self.memory_path
                        ))
                        sess.run(tf.global_variables_initializer())

                else:
                    sess.run(tf.global_variables_initializer())

                for i in range(iterations):
                    loss, opt_func = sess.run([cost, optimizer],
                                              feed_dict={
                                                  self.x: self.logits,
                                                  self.y: self.labels,
                                                  self.train: True
                                              })
                    print("Loss {1}: {0}".format(loss, i))

                    if i % save_checkpoint == 0:
                        print("Saved Session")
                        tf.train.Saver().save(sess, self.memory_path)
            tf.reset_default_graph()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()

    @staticmethod
    def make_prediction(img_array):
        """Makes a prediction based off of an image array

        :param img_array: cv2/np.array of a image to be predicted

        :returns: dictionary of prediction list, and probability list
            For Example:
                {"prediction": prediction, "probability":probability}

        """
        with CryptoNeuralNetwork() as self:
            img_array = [img_array]
            prediction = self.model(self.x)
            probability = tf.nn.softmax(prediction)
            with tf.Session() as sess:
                tf.train.Saver().restore(sess, self.memory_path)
                prediction, probability = sess.run([prediction, probability],
                                                   feed_dict={
                                                       self.x: img_array,
                                                       self.train: False
                                                   })
                prediction = list(prediction[0])
                prediction = np.argmax(prediction)

                prediction = self.directories[prediction]
                probability = list(probability[0])

                return {
                    "prediction": prediction,
                    "probability": probability
                }


def make_test_data(func):
    x = np.linspace(0,4,100)
    y = func(np.linspace(0,4,100))
    GenData.create_directory("testDirectory/")
    GenData.save_img(os.path.join("testDirectory/", func.__name__ + ".jpeg"), x, y)
