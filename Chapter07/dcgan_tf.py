'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 7 Generative Adversarial Networks
Author: Yuxi (Hayden) Liu
'''
import numpy as np
import tensorflow as tf


def load_dataset_pad():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./mnist_data')
    train_data = np.concatenate((x_train, x_test), axis=0)
    train_data = train_data / 255.
    train_data = train_data * 2. - 1
    train_data = train_data.reshape([-1, 28, 28, 1])
    train_data = np.pad(train_data, ((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values=0.)
    return train_data



def gen_batches(data, batch_size, shuffle=True):
    """
    Generate batches for training
    @param data: training data
    @param batch_size: batch size
    @param shuffle: shuffle the data or not
    @return: batches generator
    """
    n_data = data.shape[0]
    if shuffle:
        idx = np.arange(n_data)
        np.random.shuffle(idx)
        data = data[idx]

    for i in range(0, n_data, batch_size):
        batch = data[i:i + batch_size]
        yield batch



data = load_dataset_pad()
print("Training dataset shape:", data.shape)


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


display_images(data, 32)


def dense(x, n_outputs, activation=None):
    return tf.layers.dense(x, n_outputs, activation=activation,
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))


def conv2d(x, n_filters, kernel_size=5):
    return tf.layers.conv2d(inputs=x, filters=n_filters, kernel_size=kernel_size, strides=2, padding="same",
                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

def transpose_conv2d(x, n_filters, kernel_size=5):
    return tf.layers.conv2d_transpose(inputs=x, filters=n_filters, kernel_size=kernel_size, strides=2, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))


def batch_norm(x, training, epsilon=1e-5, momentum=0.9):
    return tf.layers.batch_normalization(x, training=training, epsilon=epsilon, momentum=momentum)


def generator(z, n_channel, training=True):
    """
    Generator network for DCGAN
    @param z: input of random samples
    @param n_channel: number of output channels
    @param training: whether to return the output in training mode (normalized with statistics of the current batch)
    @return: output of the generator network
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        fc = dense(z, 256 * 4 * 4, activation=tf.nn.relu)
        fc = tf.reshape(fc, (-1, 4, 4, 256))

        trans_conv1 = transpose_conv2d(fc, 128)
        trans_conv1 = batch_norm(trans_conv1, training=training)
        trans_conv1 = tf.nn.relu(trans_conv1)

        trans_conv2 = transpose_conv2d(trans_conv1, 64)
        trans_conv2 = batch_norm(trans_conv2, training=training)
        trans_conv2 = tf.nn.relu(trans_conv2)

        trans_conv3 = transpose_conv2d(trans_conv2, n_channel)
        out = tf.tanh(trans_conv3)
        return out



def discriminator(x, alpha=0.2, training=True):
    """
    Discriminator network for DCGAN
    @param x: input samples, can be real or generated samples
    @param alpha: leaky relu factor
    @param training: whether to return the output in training mode (normalized with statistics of the current batch)
    @return: output logits
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        conv1 = conv2d(x, 64)
        conv1 = tf.nn.leaky_relu(conv1, alpha)

        conv2 = conv2d(conv1, 128)
        conv2 = batch_norm(conv2, training=training)
        conv2 = tf.nn.leaky_relu(conv2, alpha)

        conv3 = conv2d(conv2, 256)
        conv3 = batch_norm(conv3, training=training)
        conv3 = tf.nn.leaky_relu(conv3, alpha)

        fc = tf.layers.flatten(conv3)
        out = dense(fc, 1)

        return out




image_size = data.shape[1:]
noise_size = 100
learning_rate = 0.0002
batch_size = 128
epochs = 50
alpha = 0.2
beta1 = 0.5


tf.reset_default_graph()

X_real = tf.placeholder(tf.float32, (None,) + image_size, name='input_real')

z = tf.placeholder(tf.float32, (None, noise_size), name='input_noise')


g_sample = generator(z, image_size[2])

d_real_out = discriminator(X_real)
d_fake_out = discriminator(g_sample)


g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_out, labels=tf.ones_like(d_fake_out)))
tf.summary.scalar('generator_loss', g_loss)

d_real_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_out,
                                            labels=tf.ones_like(d_real_out)))

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


steps = 0
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logdir/dcgan', sess.graph)

    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for batch_x in gen_batches(data, batch_size):

            batch_z = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            _, summary, d_loss_batch = sess.run([d_opt, merged, d_loss], feed_dict={z: batch_z, X_real: batch_x})

            sess.run(g_opt, feed_dict={z: batch_z, X_real: batch_x})
            _, g_loss_batch = sess.run([g_opt, g_loss], feed_dict={z: batch_z, X_real: batch_x})

            if steps % 100 == 0:
                train_writer.add_summary(summary, steps)
                print("Epoch {}/{} - discriminator loss: {:.4f}, generator Loss: {:.4f}".format(
                    epoch + 1, epochs, d_loss_batch, g_loss_batch))


            steps += 1

        gen_samples = sess.run(generator(z, image_size[2], training=False), feed_dict={z: sample_z})

        display_images(gen_samples, 32)





