'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 7 Generative Adversarial Networks
Author: Yuxi (Hayden) Liu
'''
import numpy as np
import tensorflow as tf


def load_dataset_label():
    from keras.utils import np_utils
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./mnist_data')
    x_data = np.concatenate((x_train, x_test), axis=0)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_data = np.concatenate((y_train, y_test), axis=0)
    x_data = x_data / 255.
    x_data = x_data * 2. - 1
    x_data = x_data.reshape([-1, 28 * 28])
    return x_data, y_data


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


x_data, y_data = load_dataset_label()
print("Training dataset shape:", x_data.shape)

import matplotlib.pyplot as plt

def display_images(data, image_size=28):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        img = data[i, :]
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img.reshape(image_size, image_size), cmap='gray')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


display_images(x_data)


def dense(x, n_outputs, activation=None):
    return tf.layers.dense(x, n_outputs, activation=activation,
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))


def generator(z, y, alpha=0.2):
    """
    Generator network for CGAN
    @param z: input of random samples
    @param y: labels of the input samples
    @param alpha: leaky relu factor
    @return: output of the generator network
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        z_y = tf.concat([z, y], axis=1)
        fc1 = dense(z_y, 256)
        fc1 = tf.nn.leaky_relu(fc1, alpha)

        fc2 = dense(fc1, 512)
        fc2 = tf.nn.leaky_relu(fc2, alpha)

        fc3 = dense(fc2, 1024)
        fc3 = tf.nn.leaky_relu(fc3, alpha)

        out = dense(fc3, 28 * 28)
        out = tf.tanh(out)
        return out


def discriminator(x, y, alpha=0.2):
    """
    Discriminator network for CGAN
    @param x: input samples, can be real or generated samples
    @param y: labels of the input samples
    @param alpha: leaky relu factor
    @return: output logits
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x_y = tf.concat([x, y], axis=1)
        fc1 = dense(x_y, 1024)
        fc1 = tf.nn.leaky_relu(fc1, alpha)

        fc2 = dense(fc1, 512)
        fc2 = tf.nn.leaky_relu(fc2, alpha)

        fc3 = dense(fc2, 256)
        fc3 = tf.nn.leaky_relu(fc3, alpha)

        out = dense(fc3, 1)
        return out


noise_size = 100
learning_rate = 0.0002
batch_size = 128
epochs = 100
alpha = 0.2
beta1 = 0.5

tf.reset_default_graph()
## Real Input
X_real = tf.placeholder(tf.float32, (None, 28 * 28), name='input_real')
# Latent Variables / Noise
z = tf.placeholder(tf.float32, (None, noise_size), name='input_noise')

n_classes = 10
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_classes')

g_sample = generator(z, y)

d_real_out = discriminator(X_real, y)
d_fake_out = discriminator(g_sample, y)


g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_out, labels=tf.ones_like(d_fake_out)))

tf.summary.scalar('generator_loss', g_loss)


d_real_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_out, labels=tf.ones_like(d_real_out)))


d_fake_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_out, labels=tf.zeros_like(d_fake_out)))

d_loss = d_real_loss + d_fake_loss

tf.summary.scalar('discriminator_loss', d_loss)

train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if var.name.startswith('discriminator')]
g_vars = [var for var in train_vars if var.name.startswith('generator')]


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

n_sample_display = 40
sample_z = np.random.uniform(-1, 1, size=(n_sample_display, noise_size))

sample_y = np.zeros(shape=(n_sample_display, n_classes))

for i in range(n_sample_display):
    j = i % 10
    sample_y[i, j] = 1


steps = 0
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logdir/cgan', sess.graph)

    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for batch_x, batch_y in gen_batches_label(x_data, y_data, batch_size):

            batch_z = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            _, summary, d_loss_batch = sess.run([d_opt, merged, d_loss], feed_dict={z: batch_z, X_real: batch_x, y: batch_y})

            sess.run(g_opt, feed_dict={z: batch_z, y: batch_y})
            _, g_loss_batch = sess.run([g_opt, g_loss], feed_dict={z: batch_z, y: batch_y})

            if steps % 100 == 0:
                train_writer.add_summary(summary, steps)
                print("Epoch {}/{} - discriminator loss: {:.4f}, generator Loss: {:.4f}".format(
                    epoch + 1, epochs, d_loss_batch, g_loss_batch))

            steps += 1
        gen_samples = sess.run(generator(z, y),
                               feed_dict={z: sample_z, y: sample_y})

        display_images(gen_samples)