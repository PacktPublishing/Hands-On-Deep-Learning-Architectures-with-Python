'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 8 New Trends of Deep Learning
Author: Yuxi (Hayden) Liu
'''

import numpy as np
import tensorflow as tf

from edward.models import Categorical, Normal
import edward as ed



def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./mnist_data')
    x_train = x_train / 255.
    x_train = x_train.reshape([-1, 28 * 28])
    x_test = x_test / 255.
    x_test = x_test.reshape([-1, 28 * 28])
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_dataset()


def gen_batches_label(x_data, y_data, batch_size, shuffle=True):
    """
    Generate batches including label for training
    @param x_data: training data
    @param y_data: training label
    @param batch_size: batch size
    @param shuffle: shuffle the data or not
    @return: batches generator
    """
    n_data = x_data.shape[0]
    if shuffle:
        idx = np.arange(n_data)
        np.random.shuffle(idx)
        x_data = x_data[idx]
        y_data = y_data[idx]
    for i in range(0, n_data - batch_size, batch_size):
        x_batch = x_data[i:i + batch_size]
        y_batch = y_data[i:i + batch_size]
        yield x_batch, y_batch



batch_size = 100
n_features = 28 * 28
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_features])

w = Normal(loc=tf.zeros([n_features, n_classes]), scale=tf.ones([n_features, n_classes]))

b = Normal(loc=tf.zeros(n_classes), scale=tf.ones(n_classes))

y = Categorical(tf.matmul(x, w) + b)

qw = Normal(loc=tf.Variable(tf.random_normal([n_features, n_classes])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_features, n_classes]))))
qb = Normal(loc=tf.Variable(tf.random_normal([n_classes])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_classes]))))


y_ph = tf.placeholder(tf.int32, [batch_size])

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})


inference.initialize(n_iter=100, scale={y: float(x_train.shape[0]) / batch_size})



sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


for _ in range(inference.n_iter):
    for X_batch, Y_batch in gen_batches_label(x_train, y_train, batch_size):
        inference.update(feed_dict={x: X_batch, y_ph: Y_batch})



# Generate samples the posterior and store them.
n_samples = 30
pred_samples = []

for _ in range(n_samples):
    w_sample = qw.sample()
    b_sample = qb.sample()
    prob = tf.nn.softmax(tf.matmul(x_test.astype(np.float32), w_sample) + b_sample)
    pred = np.argmax(prob.eval(), axis=1).astype(np.float32)
    pred_samples.append(pred)



acc_samples = []
for pred in pred_samples:
    acc = (pred == y_test).mean() * 100
    acc_samples.append(acc)

print('The classification accuracy for each sample of w and b:', acc_samples)

image_test_ind = 0
image_test = x_test[image_test_ind]
label_test = y_test[image_test_ind]
print('The label of the image is:', label_test)

import matplotlib.pyplot as plt
plt.imshow(image_test.reshape((28, 28)), cmap='Blues')
plt.show()


pred_samples_test = [pred[image_test_ind] for pred in pred_samples]
print('The predictions for the example are:', pred_samples_test)

plt.hist(pred_samples_test, bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0, 10)
plt.xlabel("Predictions for the example")
plt.ylabel("Frequency")
plt.show()


from scipy import ndimage
image_file = 'notMNIST_small/A/MDRiXzA4LnR0Zg==.png'
image_not = ndimage.imread(image_file).astype(float)

plt.imshow(image_not, cmap='Blues')
plt.show()


image_not = image_not / 255.
image_not = image_not.reshape([-1, 28 * 28])


pred_samples_not = []

for _ in range(n_samples):
    w_sample = qw.sample()
    b_sample = qb.sample()
    prob = tf.nn.softmax(tf.matmul(image_not.astype(np.float32), w_sample) + b_sample)
    pred = np.argmax(prob.eval(), axis=1).astype(np.float32)
    pred_samples_not.append(pred[0])


print('The predictions for the notMNIST example are:', pred_samples_not)

plt.hist(pred_samples_not, bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0,10)
plt.xlabel("Predictions for the notMNIST example")
plt.ylabel("Frequency")
plt.show()