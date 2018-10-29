#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BayesSCDC model for MNIST without annotations, i.e., unsupervised clustering.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

from six.moves import range, zip
import tensorflow as tf
import numpy as np
import zhusuan as zs
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

from utils import dataset, setup_logger, save_image_collections, cluster_acc
from distributions import niw, catgorical, mvn, dirichlet
from distributions import normalize, exp_family_kl


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("seed", 1234, """Random seed.""")


def get_global_params(scope, d, K, alpha, niw_conc, random_scale=None,
                      trainable=True):
    def init_niw_param():
        # nu: scalar, S: (d, d), m: (d,) kappa: scalar
        # TODO: nu different to orig code, different init of nu, S?
        nu, S, m, kappa = (tf.constant(d + niw_conc, dtype=tf.float32),
                           (d + niw_conc) * tf.eye(d),
                           tf.zeros(d),
                           tf.constant(niw_conc, dtype=tf.float32))
        if random_scale:
            m = m + random_scale * tf.random_normal(m.shape)
        return niw.standard_to_natural(m, kappa, S, nu)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if random_scale:
            dir_nat = tf.random_uniform([K], minval=0, maxval=alpha - 1.)
        else:
            dir_nat = tf.ones([K]) * (alpha - 1.)
        # [K]
        dir_params = tf.get_variable(
            "dir_params", dtype=tf.float32, initializer=dir_nat,
            trainable=trainable)
        # [K, d + d^2 + 2]
        niw_nat = tf.stack([init_niw_param() for _ in range(K)])
        niw_params = tf.get_variable(
            "niw_params", dtype=tf.float32, initializer=niw_nat,
            trainable=trainable)
    return dir_params, niw_params


def global_expected_stats(global_params, d):
    dir_params, niw_params = global_params
    # [K]
    dir_stats = dirichlet.expected_stats(dir_params)
    # [K, d + d^2 + 2]
    niw_stats = niw.expected_stats(niw_params, d)
    return dir_stats, niw_stats


@zs.reuse("encoder")
def encoder(o, d):
    h = tf.layers.dense(tf.to_float(o), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_param_1 = tf.layers.dense(h, d)
    x_sigma_inv = tf.layers.dense(h, d, activation=tf.nn.softplus)
    x_param_2 = -0.5 * x_sigma_inv
    return x_param_1, x_param_2


@zs.reuse("decoder")
def decoder(x, o_dim):
    h = tf.layers.dense(x, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    o_logits = tf.layers.dense(h, o_dim)
    o_dist = zs.distributions.Bernoulli(o_logits, group_ndims=1)
    return o_dist, tf.sigmoid(o_logits)


def x_mean_field(niw_stats, z_stats, x_obs_param, d):
    # niw_stats: [K, d + d^2 + 2], z_stats: [M, K]
    # x_prior_term: [M, d + d^2]
    x_prior_term = tf.matmul(z_stats, niw_stats[:, :-2])
    # x_obs_param: [M, d + d^2]
    # x_nat_param: [M, d + d^2]
    x_nat_param = x_prior_term + x_obs_param
    # x_stats: [M, d + d^2]
    x_stats = mvn.expected_stats(x_nat_param, d)
    return x_nat_param, x_stats


def z_mean_field(global_stats, x_stats):
    # dir_stats: [K]
    dir_stats, niw_stats = global_stats
    # x_stats: [M, d + d^2], niw_stats: [K, d + d^2 + 2]
    M = tf.shape(x_stats)[0]
    # x_stats_pad: [M, d + d^2 + 2]
    x_stats_pad = tf.concat([x_stats, tf.ones([M, 2])], axis=-1)
    # z_nat_param: [M, K]
    z_nat_param = dir_stats + tf.matmul(x_stats_pad, niw_stats,
                                        transpose_b=True)
    # z_stats: [M, K]
    z_stats = catgorical.expected_stats(z_nat_param)
    return z_nat_param, z_stats


def local_kl_z(z_nat_param, dir_stats, z_stats):
    # z_nat_param: [M, K]
    # dir_stats: [K]
    # z_stats: [M, K]
    z_nat_param = z_nat_param - tf.reduce_logsumexp(z_nat_param, axis=-1,
                                                    keepdims=True)
    nat_param_diff = z_nat_param - dir_stats
    # ret: [M]
    return exp_family_kl(nat_param_diff, z_stats)


def local_kl_x(x_nat_param, niw_stats, z_stats, x_stats, d):
    # x_nat_param: [M, d + d^2]
    # niw_stats: [K, d + d^2 + 2]
    # z_stats: [M, K]
    # x_stats: [M, d + d^2]
    # x_prior_term: [M, d + d^2 + 2]
    x_prior_term = tf.matmul(z_stats, niw_stats)
    # nat_param_diff: [M, d + d^2]
    nat_param_diff = x_nat_param - x_prior_term[:, :-2]
    # log_partition_diff: [M]
    log_z_diff = mvn.log_partition(x_nat_param, d) + tf.reduce_sum(
        x_prior_term[:, -2:], axis=-1)
    # ret: [M]
    return exp_family_kl(nat_param_diff, x_stats, log_z_diff=log_z_diff)


def global_kl(prior_global_params, global_params, global_stats, d):
    prior_dir_param, prior_niw_param = prior_global_params

    def _kl_helper(log_partition, param, prior_param, stats):
        nat_diff = param - prior_param
        log_z_diff = log_partition(param) - log_partition(prior_param)
        return exp_family_kl(nat_diff, stats, log_z_diff=log_z_diff)

    # dir_param: [K], niw_param: [K, d + d^2 + 2]
    dir_param, niw_param = global_params
    # dir_stats: [K], niw_stats: [K, d + d^2 + 2]
    dir_stats, niw_stats = global_stats
    # dir_kl: []
    dir_kl = _kl_helper(dirichlet.log_partition, dir_param, prior_dir_param,
                        dir_stats)
    # niw_kl: [K]
    niw_kl = _kl_helper(lambda x: niw.log_partition(x, d), niw_param,
                        prior_niw_param, niw_stats)
    return dir_kl + tf.reduce_sum(niw_kl, axis=0)


def elbo(log_po_term, local_kl_z, local_kl_x, global_kl, N):
    # log_po_term: [M]
    # local_kl_z: [M], local_kl_x: [M]
    # global_kl: []
    obj = tf.reduce_mean(log_po_term - local_kl_z - local_kl_x) - global_kl / N
    # ret: []
    return obj


def variational_message_passing(prior_global_params, global_params,
                                o, o_dim, d, K, N, n_iters=100):
    global_stats = global_expected_stats(global_params, d)
    dir_stats, niw_stats = global_stats
    M = tf.shape(o)[0]
    z_stats = normalize(tf.random_uniform([M, K], 1e-8, maxval=1))
    # h: [M, d], J: [M, d]
    h, J = encoder(o, d)
    # J: [M, d * d]
    J = tf.reshape(tf.matrix_diag(J), [M, d * d])
    # x_obs_param: [M, d + d * d]
    x_obs_param = tf.concat([h, J], axis=-1)
    for t in range(n_iters):
        x_nat_param, x_stats = x_mean_field(niw_stats, z_stats, x_obs_param, d)
        z_nat_param, z_stats = z_mean_field(global_stats, x_stats)
    # x: [M, d]
    x = mvn.sample(x_nat_param, d)
    o_dist, _ = decoder(x, o_dim)
    # log_po_term: [M]
    log_po_term = o_dist.log_prob(o)
    # log_kl_x_term: [M]
    local_kl_x_term = local_kl_x(x_nat_param, niw_stats, z_stats, x_stats, d)
    # log_kl_z_term: [M]
    local_kl_z_term = local_kl_z(z_nat_param, dir_stats, z_stats)
    # global_kl_term: []
    global_kl_term = global_kl(
        prior_global_params, global_params, global_stats, d)
    lower_bound = elbo(log_po_term, local_kl_z_term, local_kl_x_term,
                       global_kl_term, N)
    # Natural gradient for global variational parameters
    # z_stats: [M, K], x_stats: [M, d + d^2]
    # dir_updates: [K]
    dir_updates = tf.reduce_mean(z_stats, axis=0)
    # niw_updates: [K, d + d^2 + 2]
    niw_updates = tf.matmul(z_stats, tf.concat([x_stats, tf.ones([M, 2])], -1),
                            transpose_a=True) / tf.to_float(M)
    updates = (dir_updates, niw_updates)
    nat_grads = [(prior_global_params[i] - global_params[i]) / N + updates[i]
                 for i in range(2)]
    return lower_bound, nat_grads, z_stats, dir_stats, niw_stats


def main():
    seed = FLAGS.seed
    result_path = "results/mnist_{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), seed)
    logger = setup_logger('mnist', __file__, result_path)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Load MNIST
    data_path = os.path.join('data', 'mnist.pkl.gz')
    o_train, t_train, o_valid, t_valid, o_test, t_test = \
        dataset.load_mnist_realval(data_path, one_hot=False)
    o_train = np.vstack([o_train, o_valid])
    t_train = np.hstack([t_train, t_valid])
    o_test = np.random.binomial(1, o_test, size=o_test.shape)
    n_train, o_dim = o_train.shape
    n_test, _ = o_test.shape
    # n_class = np.max(t_test) + 1

    # Prior parameters
    d = 8
    K = 50
    prior_alpha = 1.05
    prior_niw_conc = 0.5

    # Variational initialization
    alpha = 2.
    niw_conc = 1.
    random_scale = 3.

    # learning rate
    learning_rate = 1e-3
    nat_grad_scale = 1e4

    prior_global_params = get_global_params("prior", d, K, prior_alpha,
                                            prior_niw_conc, trainable=False)
    global_params = get_global_params("variational", d, K, alpha, niw_conc,
                                      random_scale=random_scale, trainable=True)

    # n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    o_input = tf.placeholder(tf.float32, shape=[None, o_dim], name='o')
    o = tf.to_int32(tf.random_uniform(tf.shape(o_input)) <= o_input)

    lower_bound, global_nat_grads, z_stats, dir_stats, niw_stats = \
        variational_message_passing(prior_global_params, global_params,
                                    o, o_dim, d, K, n_train, n_iters=4)
    z_pred = tf.argmax(z_stats, axis=-1)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9)
    net_vars = (tf.trainable_variables(scope="encoder") +
                tf.trainable_variables(scope="decoder"))
    net_grads_and_vars = optimizer.compute_gradients(
        -lower_bound, var_list=net_vars)
    global_nat_grads = [-nat_grad_scale * g for g in global_nat_grads]
    global_grads_and_vars = list(zip(global_nat_grads, global_params))
    infer_op = optimizer.apply_gradients(net_grads_and_vars +
                                         global_grads_and_vars)

    # Generation
    # niw_stats: [K, d + d^2 + 2]
    gen_mvn_params = niw_stats[:, :-2]
    # transparency: [K]
    transp = tf.exp(dir_stats) / tf.reduce_max(tf.exp(dir_stats))
    # x_samples: [K, d, 10]
    x_samples = mvn.sample(gen_mvn_params, d, n_samples=10)
    # o_mean: [10, K, o_dim]
    _, o_mean = decoder(tf.transpose(x_samples, [2, 0, 1]), o_dim)
    # o_gen: [10 * K, 28, 28, 1]
    o_gen = tf.reshape(o_mean * transp[:, None], [-1, 28, 28, 1])

    # Define training parameters
    epochs = 200
    batch_size = 128
    iters = o_train.shape[0] // batch_size
    save_freq = 1
    test_freq = 10
    test_batch_size = 400
    test_iters = o_test.shape[0] // test_batch_size

    def _evaluate(pred_batches, labels):
        preds = np.hstack(pred_batches)
        truths = labels[:preds.size]
        acc, _ = cluster_acc(preds, truths)
        nmi = adjusted_mutual_info_score(truths, labels_pred=preds)
        return acc, nmi

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            indices = np.random.permutation(n_train)
            o_train = o_train[indices]
            t_train = t_train[indices]
            lbs = []
            t_preds = []
            for t in range(iters):
                o_batch = o_train[t * batch_size:(t + 1) * batch_size]
                _, lb, t_pred = sess.run(
                    [infer_op, lower_bound, z_pred],
                    feed_dict={o_input: o_batch})
                # print("lb: {}".format(lb))
                lbs.append(lb)
                t_preds.append(t_pred)

            time_epoch += time.time()
            train_acc, train_nmi = _evaluate(t_preds, t_train)
            logger.info(
                'Epoch {} ({:.1f}s): Lower bound = {}, acc = {}, nmi = {}'
                .format(epoch, time_epoch, np.mean(lbs), train_acc, train_nmi))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_t_preds = []
                for t in range(test_iters):
                    test_o_batch = o_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb, test_t_pred = sess.run([lower_bound, z_pred],
                                                    feed_dict={o: test_o_batch})
                    test_lbs.append(test_lb)
                    test_t_preds.append(test_t_pred)

                time_test += time.time()
                test_acc, test_nmi = _evaluate(test_t_preds, t_test)
                logger.info('>>> TEST ({:.1f}s)'.format(time_test))
                logger.info('>> Test lower bound = {}, acc = {}, nmi = {}'
                            .format(np.mean(test_lbs), test_acc, test_nmi))

                if epoch == epochs:
                    with open('results/mnist_bayesSCDC_unsup.txt', "a") as myfile:
                        myfile.write("seed: %d train_acc: %f train_nmi: %f "
                                     "test_acc: %f test_nmi: %f" % (
                            seed, train_acc, train_nmi, test_acc, test_nmi))
                        myfile.write('\n')
                        myfile.close()

            if epoch % save_freq == 0:
                logger.info('Saving images...')
                images = sess.run(o_gen)
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name, shape=(10, K))


if __name__ == "__main__":
    main()
