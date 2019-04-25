# CIFAR-10 image classification

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf
from sklearn.utils import shuffle
np.set_printoptions(threshold = sys.maxsize)

# define the path to the directory where you have extracted the zipped data
DATA_DIR = 'cifar-10-batches-py'

#hyper-parameters
BATCH_SIZE = 128
CLASS_NUM = 10
EPOCHS = 20
DROPOUT = 0.5
LEARNING_RATE = 0.001
IMAGE_SIZE = (32, 32)
SEED = 2 

# function to load data

class data:

	def __init__(self, dataDir, fileName, batchSize, seed, classNum = 10):

		self.dataDir = dataDir
		self.fileName = fileName
		self.classNum = classNum
		self.batchSize = batchSize
		self.seed = seed

		self.labelsDicti = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
		self.inverseLabelsDicti = {v:k for k,v in self.labelsDicti.items()}

	def load_data_batch(self):

		with open(os.path.join(self.dataDir, self.fileName), 'rb') as f:

			dataBatch = pickle.load(f, encoding = 'latin1') 
			#print(dataBatch['data'].shape)
			# latin1 encoding has been used to dump the data.

			# we don't need filename and other details,
			# we will keep only labels and images

		self.images = dataBatch['data']
		self.labels = dataBatch['labels']

	def reshape_data(self):

		# function to reshape and transpose
		self.images = self.images.reshape(len(self.images), 3, 32, 32).transpose(0, 2, 3, 1)

	def visualise_data(self, indices):

		plt.figure(figsize = (5, 5))

		for i in range(len(indices)):
			# take out the ith image in indices 
			img = self.images[indices[i]]

			# it's corresponding label
			label =self.labels[indices[i]]

			plt.subplot(2,2,i+1)
			plt.imshow(img)
			plt.title(self.labelsDicti[label])

		plt.show()

	def one_hot_encoder(self):

		# this function will convert the labels into one-hot vectors
		# intially the label vector is a list, we will convert it to numpy array,

		self.labels = np.array(self.labels, dtype = np.int32)

		#converting to one-hot
		self.labels = np.eye(self.classNum)[self.labels]

		#print(self.labels.shape)

	def normalize_images(self):

		# just simply dividing by 255
		self.images = self.images / 255

	def shuffle_data(self):

		# shuffle the data so that training is better
		self.images, self.labels = shuffle(self.images, self.labels, random_state = self.seed)


	def generate_batches(self):

		# function to yield out batches of batchSize from the loaded file
		for i in range(0, len(self.images), self.batchSize):

			last = min(i + self.batchSize, len(self.images))

			#yield(np.random.rand(128, 32, 32, 3), self.labels[i:last])
			yield (self.images[i: last], self.labels[i: last])

class model:

	def __init__(self, batchSize, classNum, dropOut, learningRate, epochs, imageSize, savePath ):

		self.batchSize = batchSize
		self.classNum = classNum
		self.dropOut = dropOut
		self.imageSize = imageSize

		self.learningRate = learningRate
		self.epochs = epochs
		self.savePath = savePath


		# we will define model architecture here so that it get's initialize as 
		# soon as we call the class
		
		with tf.name_scope('placeholders') as scope:

			# making placeholders for inputs (x) and labels (y)
			self.x = tf.placeholder(shape = [None, self.imageSize[0], self.imageSize[1], 3], dtype = tf.float32, name = 'inp_x')
			self.y = tf.placeholder(shape = [None, self.classNum], dtype = tf.float32, name = 'true_y')
			self.keepProb = tf.placeholder(tf.float32)

		#first conv layer with 64 filters
		with tf.name_scope('conv_1') as scope:

			#tensorflow takes the kernel as a 4-D tensor. We can initialize the values with 
			# truncated normal distribution.
			filter1 = tf.Variable(tf.zeros([3, 3, 3, 64], dtype=tf.float32), name='filter_1')

			conv1 = tf.nn.relu(tf.nn.conv2d(self.x, filter1, [1, 1, 1, 1], padding='SAME', name = 'convo_1'))


		with tf.name_scope('pool_1') as scope:

			pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],padding='SAME', name = 'maxPool_1')


		with tf.name_scope('conv_2') as scope:

			filter2 = tf.Variable(tf.zeros([2, 2, 64, 128], dtype=tf.float32), name='filter_2')

			conv2 = tf.nn.relu(tf.nn.conv2d(pool1, filter2, [1, 1, 1, 1], padding='SAME', name = 'convo_2'))


		with tf.name_scope('conv_3') as scope:

			filter3 = tf.Variable(tf.zeros([2, 2, 128, 128], dtype=tf.float32), name='filter_3')

			conv3 = tf.nn.relu(tf.nn.conv2d(conv2, filter3, [1, 1, 1, 1], padding='SAME', name = 'convo_3'))


		with tf.name_scope('pool_2') as scope:

			pool2 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
									 padding='SAME', name = 'maxPool_2')


		with tf.name_scope('conv_4') as scope:

			filter4 = tf.Variable(tf.zeros([1, 1, 128, 256], dtype=tf.float32), name='filter_4')

			conv4 = tf.nn.relu(tf.nn.conv2d(pool2, filter4, [1, 1, 1, 1], padding='SAME', name = 'convo_4'))


		with tf.name_scope('pool_3') as scope:

			pool3 = tf.nn.max_pool(conv4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
									 padding='SAME', name = 'maxPool_3')


		with tf.name_scope('conv_5') as scope:

			filter5 = tf.Variable(tf.zeros([1, 1, 256, 512], dtype=tf.float32), name='filter_5')

			conv5 = tf.nn.relu(tf.nn.conv2d(pool3, filter5, [1, 1, 1, 1], padding='SAME', name = 'convo_5'))


		with tf.name_scope('flatten') as scope:

			flatt = tf.layers.Flatten()(conv5)
			shape = conv5.get_shape().as_list()
			print(shape)
			flatt = tf.reshape(conv5, [-1, shape[1]*shape[2]*shape[3]])


		with tf.name_scope('dense_1') as scope:

			dense1 = tf.layers.dense(flatt, units = 1024, activation = 'relu',name='fc_1')

			dropOut1 = tf.nn.dropout(dense1, self.keepProb)


		with tf.name_scope('dense_2') as scope:

			dense2 = tf.layers.dense(dropOut1, units = 512, activation = 'relu',name='fc_2')

			dropOut2 = tf.nn.dropout(dense2, self.keepProb)

		with tf.name_scope('dense_3') as scope:

			dense3 = tf.layers.dense(dropOut2, units = 256, activation = 'relu',name='fc_3')

			dropOut3 = tf.nn.dropout(dense3, self.keepProb)


		with tf.name_scope('out') as scope:

			outLayer = tf.layers.dense(dropOut3, units = self.classNum, activation = None, name='out_layer')


		with tf.name_scope('loss') as scope:

			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = outLayer, labels = self.y))

		with tf.name_scope('optimizer') as scope:

			optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)

			self.train = optimizer.minimize(self.loss)


		with tf.name_scope('accuracy') as scope:

			correctPredictions = tf.equal(tf.argmax(outLayer, axis=1), tf.argmax(self.y, axis = 1))

			# calculating average accuracy
			self.avgAccuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

modelGraph = model(batchSize = BATCH_SIZE, classNum = CLASS_NUM, dropOut = DROPOUT,
					learningRate = LEARNING_RATE, epochs = EPOCHS, imageSize = IMAGE_SIZE, savePath = 'model')


with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	for epoch in range(modelGraph.epochs):

		for iBatch in range(1, 6):

			dataObj = data(DATA_DIR, 'data_batch_' + str(iBatch), BATCH_SIZE, SEED)
			dataObj.load_data_batch()
			dataObj.reshape_data()
			#dataObj.visualise_data([100, 4000, 2, 8000])
			dataObj.one_hot_encoder()
			dataObj.normalize_images()
			dataObj.shuffle_data()
			#print(dataObj.generate_batches()[0])

			for batchX, batchY in dataObj.generate_batches():

				#print(batchX[0])
				#print(batchY[0])

				_, lossT, accT = sess.run([modelGraph.train, modelGraph.loss, modelGraph.avgAccuracy],
								feed_dict = {modelGraph.x: batchX, modelGraph.y: batchY, modelGraph.keepProb: modelGraph.dropOut})

				print('Epoch: '+str(epoch)+' Minibatch_Loss: '+"{:.6f}".format(lossT)+' Train_acc: '+"{:.5f}".format(accT)+"\n")

			if epoch % 10 == 0:

				saver.save(sess, modelGraph.savePath)


