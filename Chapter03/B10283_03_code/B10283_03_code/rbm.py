'''
Source codes for Hands-On Deep Learning Architectures with Python (Packt Publishing)
Chapter 3 Restricted Boltzmann Machines and Autoencoders
Author: Yuxi (Hayden) Liu
'''

import numpy as np
import tensorflow as tf


class RBM(object):
    def __init__(self, num_v, num_h, batch_size, learning_rate, num_epoch, k=2):
        self.num_v = num_v
        self.num_h = num_h
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.k = k
        self.W, self.a, self.b = self._init_parameter()

    def _init_parameter(self):
        """ Initializing the model parameters including weights and bias """
        abs_val = np.sqrt(2.0 / (self.num_h + self.num_v))
        W = tf.get_variable('weights', shape=(self.num_v, self.num_h),
                            initializer=tf.random_uniform_initializer(minval=-abs_val, maxval=abs_val))
        a = tf.get_variable('visible_bias', shape=(self.num_v), initializer=tf.zeros_initializer())
        b = tf.get_variable('hidden_bias', shape=(self.num_h), initializer=tf.zeros_initializer())
        return W, a, b

    def _prob_v_given_h(self, h):
        """
        Computing conditional probability P(v|h)
        @param h: hidden layer
        @return: P(v|h)
        """
        return tf.sigmoid(tf.add(self.a, tf.matmul(h, tf.transpose(self.W))))

    def _prob_h_given_v(self, v):
        """
        Computing conditional probability P(h|v)
        @param v: visible layer
        @return: P(h|v)
        """
        return tf.sigmoid(tf.add(self.b, tf.matmul(v, self.W)))

    def _bernoulli_sampling(self, prob):
        """ Bernoulli sampling based on input probability """
        distribution = tf.distributions.Bernoulli(probs=prob, dtype=tf.float32)
        return tf.cast(distribution.sample(), tf.float32)

    def _compute_gradients(self, v0, prob_h_v0, vk, prob_h_vk):
        """
        Computing gradients of weights and bias
        @param v0: visible vector before Gibbs sampling
        @param prob_h_v0: conditional probability P(h|v) before Gibbs sampling
        @param vk: visible vector after Gibbs sampling
        @param prob_h_vk: conditional probability P(h|v) after Gibbs sampling
        @return: gradients of weights, gradients of visible bias, gradients of hidden bias
        """
        outer_product0 = tf.matmul(tf.transpose(v0), prob_h_v0)
        outer_productk = tf.matmul(tf.transpose(vk), prob_h_vk)
        W_grad = tf.reduce_mean(outer_product0 - outer_productk, axis=0)
        a_grad = tf.reduce_mean(v0 - vk, axis=0)
        b_grad = tf.reduce_mean(prob_h_v0 - prob_h_vk, axis=0)
        return W_grad, a_grad, b_grad

    def _gibbs_sampling(self, v):
        """
        Gibbs sampling
        @param v: visible layer
        @return: visible vector before Gibbs sampling, conditional probability P(h|v) before Gibbs sampling,
                 visible vector after Gibbs sampling, conditional probability P(h|v) after Gibbs sampling
        """
        v0 = v
        prob_h_v0 = self._prob_h_given_v(v0)
        vk = v
        prob_h_vk = prob_h_v0

        for _ in range(self.k):
            hk = self._bernoulli_sampling(prob_h_vk)
            prob_v_hk = self._prob_v_given_h(hk)
            vk = self._bernoulli_sampling(prob_v_hk)
            prob_h_vk = self._prob_h_given_v(vk)

        return v0, prob_h_v0, vk, prob_h_vk

    def _optimize(self, v):
        """
        Optimizing RBM model parameters
        @param v: input visible layer
        @return: updated parameters, mean squared error of reconstructing v
        """
        v0, prob_h_v0, vk, prob_h_vk = self._gibbs_sampling(v)
        W_grad, a_grad, b_grad = self._compute_gradients(v0, prob_h_v0, vk, prob_h_vk)
        para_update = [tf.assign(self.W, tf.add(self.W, self.learning_rate*W_grad)),
                       tf.assign(self.a, tf.add(self.a, self.learning_rate*a_grad)),
                       tf.assign(self.b, tf.add(self.b, self.learning_rate*b_grad))]
        error = tf.metrics.mean_squared_error(v0, vk)[1]
        return para_update, error


    def train(self, X_train):
        """
        Model training
        @param X_train: input data for training
        """
        X_train_plac = tf.placeholder(tf.float32, [None, self.num_v])

        para_update, error = self._optimize(X_train_plac)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init)
            epochs_err = []
            n_batch = int(X_train.shape[0] / self.batch_size)

            for epoch in range(1, self.num_epoch + 1):
                epoch_err_sum = 0

                for batch_number in range(n_batch):

                    batch = X_train[batch_number * self.batch_size: (batch_number + 1) * self.batch_size]

                    _, batch_err = sess.run((para_update, error), feed_dict={X_train_plac: batch})

                    epoch_err_sum += batch_err

                epochs_err.append(epoch_err_sum / n_batch)

                if epoch % 10 == 0:
                    print("Training error at epoch %s: %s" % (epoch, epochs_err[-1]))

