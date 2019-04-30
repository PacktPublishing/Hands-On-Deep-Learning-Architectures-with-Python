'''
MNIST Fashion Deep Feedforward example
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras

# making object of fashion_mnist class
fashionObj = keras.datasets.fashion_mnist

# trainX contains input images and trainY contains corresponding labels 
(trainX, trainY), (testX, testY) = fashionObj.load_data()
print('train data x shape: ', trainX.shape)
print('test data x shape:', testX.shape)

print('train data y shape: ', trainY.shape)
print('test data y shape: ', testY.shape)

# make a label dictionary to map integer labels to classes
classesDict = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal',
				6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

# let's look at some images and their labels
rows = 2
columns = 2
fig = plt.figure(figsize = (5,5))

for i in range(1, rows*columns +1):
	image = trainX[i]
	label = trainY[i]

	sub = fig.add_subplot(rows, columns, i)
	sub.set_title('Label: ' + classesDict[label])

	plt.imshow(image)
plt.show()

# values of pixels in image range from 0 to 255. It is always advisable computationally
# to keep the input values between 0 to 1. Hence we will normalize our data by dividing it 
# with maximum value of 255. Also we need to reshape the input shape from (60000, 28, 28)
# to (60000, 784)

trainX = trainX.reshape(trainX.shape[0], 784) / 255.0
testX = testX.reshape(testX.shape[0], 784) / 255.0

# next we shall split the tinraing set into validation and train. We shall reserve 10% of 
# train data for validation

trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size = 0.1, random_state =2)
# random_state is used for randomly shuffling the data.


# deciding parameters for our Deep Feedforward Model

CLASS_NUM = 10
#number of classes we need to classify

INPUT_UNITS = 784 
# no. of neurons in input layer 784, as we have 28x28 = 784 pixels in each image.
# we connect each pixel to a neuron.

HIDDEN_LAYER_1_UNITS = 256
# no of neurons in first hidden layer

HIDDEN_LAYER_2_UNITS = 128
#no. of neurons in second hidden layer

OUTPUT_LAYER_UNITS = CLASS_NUM
# no. of neurons in output layer = no. of classes we have tp classify.
# each neuron will output the probability of input belonging to the class it represents

LEARNING_RATE = 0.001
# learning rate for gradient descent. Default value is 0.001

BATCH_SIZE = 64
# we will take input data in sets of 64 images at once instead of using whole data
# for every iteration. Each set is called a batch and batch function is used to generate 
# batches of data.

NUM_BATCHES = int(trainX.shape[0] / BATCH_SIZE)
# number of mini-batches required to cover the train data

EPOCHS = 20
# number of iterations we will perform to train

# the labels till now are in integers from 0 to 9. We need to make them into one-hot
trainY = np.eye(CLASS_NUM)[trainY]
valY = np.eye(CLASS_NUM)[valY]
testY = np.eye(CLASS_NUM)[testY]

# now we shall build the model graph in Tensorflow
with tf.name_scope('placeholders') as scope:

	# making placeholders for inputs (x) and labels (y)
	x = tf.placeholder(shape = [BATCH_SIZE, 784], dtype = tf.float32, name = 'inp_x')
	y = tf.placeholder(shape = [BATCH_SIZE, CLASS_NUM], dtype = tf.float32, name = 'true_y')

with tf.name_scope('inp_layer') as scope:

	# the first set of weights will be connecting the inputs layer to first hiden layer
	# Hence, it will essentially be a matrix of shape [INPUT_UNITS, HIDDEN_LAYER_1_UNITS]

	weights1 = tf.get_variable(shape = [INPUT_UNITS, HIDDEN_LAYER_1_UNITS], dtype = tf.float32,
							name = 'weights_1')

	biases1 = tf.get_variable(shape = [HIDDEN_LAYER_1_UNITS], dtype = tf.float32,
						name = 'bias_1')

	# performing W.x + b, we rather multiply x to W in due to matrix shape constraints.
	# otherwise you can also take transpose of W and mutiply it to x
	layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights1), biases1), name = 'layer_1')
	# we use the relu activations in the 2 hidden layers

with tf.name_scope('hidden_1') as scope:

	# second set of weights between hidden layer 1 and hidden layer 2
	weights2 = tf.get_variable(shape = [HIDDEN_LAYER_1_UNITS, HIDDEN_LAYER_2_UNITS], dtype = tf.float32,
							name = 'weights_2')
	biases2 = tf.get_variable(shape = [HIDDEN_LAYER_2_UNITS], dtype = tf.float32, 
						name = 'bias_2')

	# the output of layer 1 will be fed to layer 2 (as this is Feedforward Network)
	layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights2), biases2), name = 'layer_2')

with tf.name_scope('out_layer') as scope:

	#third set of weights will be from second hidden layer to final output layer
	weights3 = tf.get_variable(shape = [HIDDEN_LAYER_2_UNITS, OUTPUT_LAYER_UNITS], dtype = tf.float32,
							name = 'weights_3')
	biases3 = tf.get_variable(shape = [OUTPUT_LAYER_UNITS], dtype = tf.float32,
							name = 'biases_3')

	# In the last layer, we should use the 'softmax' activation function to get the
	# probabilities. But we won't do so here because we will use the cross entropy loss with softmax
	# which first converts the output to probabilty with softmax
	layer3 = tf.add(tf.matmul(layer2, weights3), biases3, name = 'out_layer')

# now we shall add the loss function to graph
with tf.name_scope('loss') as scope:
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer3, labels = y))

# adding optimizer 
with tf.name_scope('optimizer') as scope:

	# we will use Adam Optimizer. It is the most widely used optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)

	# we will use this optimizer to minimize loss, i.e to train the network
	train = optimizer.minimize(loss)

with tf.name_scope('accuracy') as scope:

	# here we will check how many predictions our model is making correct by comparing the labels

	# tf.equal compares the two tensors element wise, where tf.argmax returns the index of
	# class which the prediction and label belong to.
	correctPredictions = tf.equal(tf.argmax(layer3, axis=1), tf.argmax(y, axis = 1))

	# calculating average accuracy
	avgAccuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))


# beginning Tensorflow Session to start training
with tf.Session() as sess:

	# initializing Tensorflow variables under session
	sess.run(tf.global_variables_initializer())

	for epoch in range(EPOCHS):

		for i in range(NUM_BATCHES):


			# creating batch of inputs
			batchX = trainX[i*BATCH_SIZE : (i+1)*BATCH_SIZE , :]
			batchY = trainY[i*BATCH_SIZE : (i+1)*BATCH_SIZE , :]

			# running the train operation for updating weights after every mini-batch
			_, miniBatchLoss, acc = sess.run([train, loss, avgAccuracy], feed_dict = {x: batchX, y: batchY})

			# printing accuracy and loss for every 4th training batch
			if i % 10 == 0:
				print('Epoch: '+str(epoch)+' Minibatch_Loss: '+"{:.6f}".format(miniBatchLoss)+' Train_acc: '+"{:.5f}".format(acc)+"\n")

		
		# calculating loss for validation batches
		for i in range(int(valX.shape[0] / BATCH_SIZE)):

			valBatchX = valX[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]
			valBatchY = valY[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :]

			valLoss, valAcc = sess.run([loss, avgAccuracy], feed_dict = {x: valBatchX, y: valBatchY})

			if i % 5 ==0:
				print('Validation Batch: ', i,' Val Loss: ', valLoss, 'val Acc: ', valAcc)

	# after training, testing performance on test batch

	for i in range(int(testX.shape[0] / BATCH_SIZE)):

		testBatchX = testX[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]
		testBatchY = testY[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :]

		testLoss, testAcc = sess.run([loss, avgAccuracy], feed_dict = {x: testBatchX, y: testBatchY})

		if i % 5 ==0:
			print('Test Batch: ', i,' Test Loss: ', testLoss, 'Test Acc: ', testAcc)











