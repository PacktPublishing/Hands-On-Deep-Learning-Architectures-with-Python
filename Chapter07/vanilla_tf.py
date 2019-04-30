'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 7 Generative Adversarial Networks
Author: Yuxi (Hayden) Liu
'''
import numpy as np
import tensorflow as tf


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./mnist_data')
    train_data = np.concatenate((x_train, x_test), axis=0)
    train_data = train_data / 255.
    train_data = train_data * 2. - 1
    train_data = train_data.reshape([-1, 28 * 28])
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


data = load_dataset()
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


display_images(data)


def dense(x, n_outputs, activation=None):
    return tf.layers.dense(x, n_outputs, activation=activation,
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))


def generator(z, alpha=0.2):
    """
    Generator network
    @param z: input of random samples
    @param alpha: leaky relu factor
    @return: output of the generator network
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        fc1 = dense(z, 256)
        fc1 = tf.nn.leaky_relu(fc1, alpha)

        fc2 = dense(fc1, 512)
        fc2 = tf.nn.leaky_relu(fc2, alpha)

        fc3 = dense(fc2, 1024)
        fc3 = tf.nn.leaky_relu(fc3, alpha)

        out = dense(fc3, 28 * 28)
        out = tf.tanh(out)
        return out


def discriminator(x, alpha=0.2):
    """
    Discriminator network
    @param x: input samples, can be real or generated samples
    @param alpha: leaky relu factor
    @return: output logits
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        fc1 = dense(x, 1024)
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
beta1 = 0.5

tf.reset_default_graph()

X_real = tf.placeholder(tf.float32, (None, 28 * 28), name='input_real')

z = tf.placeholder(tf.float32, (None, noise_size), name='input_noise')

g_sample = generator(z)

d_real_out = discriminator(X_real)
d_fake_out = discriminator(g_sample)


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


steps = 0
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logdir/vanilla', sess.graph)

    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for batch_x in gen_batches(data, batch_size):

            batch_z = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            _, summary, d_loss_batch = sess.run([d_opt, merged, d_loss], feed_dict={z: batch_z, X_real: batch_x})

            sess.run(g_opt, feed_dict={z: batch_z})
            _, g_loss_batch = sess.run([g_opt, g_loss], feed_dict={z: batch_z})

            if steps % 100 == 0:
                train_writer.add_summary(summary, steps)
                print("Epoch {}/{} - discriminator loss: {:.4f}, generator Loss: {:.4f}".format(
                    epoch + 1, epochs, d_loss_batch, g_loss_batch))


            steps += 1

        gen_samples = sess.run(generator(z), feed_dict={z: sample_z})

        display_images(gen_samples)