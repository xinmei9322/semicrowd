#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCDC model on MNIST.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import argparse
import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs
from utils import dataset, cluster_acc, setup_logger
import utils
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from generate_annotation import two_coin, genConstraints, mygenConstraints


def loglike_annotations(qz_logits_i, qz_logits_j, theta_worker, anns, bs=100):
    """

    :param qz_logits_i: qz_logits for annotated data i. (bs, num_cluster)
    :param qz_logits_j: qz_logits for annotated data j. (bs, num_cluster)
    :param theta_worker: prior parameters list for workers.
    :param anns: (num_pairs_this_batch, 4)
    :return: log p(L|Z) scalar
    """
    alpha_logits, beta_logits = theta_worker
    logalpha = tf.log(tf.sigmoid(alpha_logits))
    logbeta = tf.log(tf.sigmoid(beta_logits))
    log_1malpha_over_beta = -alpha_logits + logalpha - logbeta  # (M, )
    log_1mbeta_over_beta = -beta_logits
    coeff_inner = alpha_logits + beta_logits

    i, j, m, labels = anns[:, 0], anns[:, 1], anns[:, 2], anns[:, 3]
    phi_i = tf.nn.softmax(qz_logits_i)  # (bs, num_cluster)
    phi_j = tf.nn.softmax(qz_logits_j)
    inner_phi_ij = tf.reduce_sum(phi_i * phi_j, axis=-1)  # (bs,)
    coeff_inner_m = tf.gather(coeff_inner, m)
    logbeta_m = tf.gather(logbeta, m)
    log_1malpha_over_beta_m = tf.gather(log_1malpha_over_beta, m)
    log_1mbeta_over_beta_m = tf.gather(log_1mbeta_over_beta, m)
    log_l = tf.to_float(labels)*(inner_phi_ij*coeff_inner_m +
                                 log_1mbeta_over_beta_m) + \
        inner_phi_ij * log_1malpha_over_beta_m + logbeta_m
    log_l = tf.reduce_mean(log_l, axis=0)
    return log_l


# relational
def init_worker(scope, num_worker, alpha=0.9, beta=0.9):
    with tf.variable_scope(scope):
        alpha_logits = tf.Variable([np.log(alpha / (1. - alpha))] * num_worker,
                                   dtype=tf.float32, name='alpha_logits')
        beta_logits = tf.Variable([np.log(beta / (1. - beta))] * num_worker,
                                  dtype=tf.float32, name='beta_logits')
        return [alpha_logits, beta_logits]


@zs.reuse('model')
def vae(observed, n, n_x, n_h, n_z):
    with zs.BayesianNet(observed=observed) as model:
        pi = tf.get_variable(name='pi', dtype=tf.float32,
                             initializer=tf.truncated_normal([1, n_z], stddev=0.1))
        zpi = tf.tile(pi, [n, 1])
        # zpi = tf.zeros([n, n_z])
        z = zs.OnehotDiscrete('z', zpi, group_ndims=0)
        h_mean = layers.fully_connected(tf.to_float(z), n_h, activation_fn=None)
        h_logstd = layers.fully_connected(tf.to_float(z), n_h, activation_fn=None)
        h = zs.Normal('h', h_mean, logstd=h_logstd, group_ndims=1)
        lx_h = layers.fully_connected(h, 512)
        lx_h = layers.fully_connected(lx_h, 512)
        x_logits = layers.fully_connected(lx_h, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model, x_logits, pi


@zs.reuse('qh')
def q_net(x, n_h):
    with zs.BayesianNet() as qh:
        lz_x = layers.fully_connected(tf.to_float(x), 512)
        lz_x = layers.fully_connected(lz_x, 512)
        h_mean = layers.fully_connected(lz_x, n_h, activation_fn=None)
        h_logstd = layers.fully_connected(lz_x, n_h, activation_fn=None)
        h = zs.Normal('h', h_mean, logstd=h_logstd, group_ndims=1)
    return qh


@zs.reuse('classifier')
def qz_net(x, n_z):
    lz_x = layers.fully_connected(tf.to_float(x), 500)
    lz_x = layers.fully_connected(lz_x, 500)
    z_logit = layers.fully_connected(lz_x, n_z, activation_fn=None)
    z = tf.nn.softmax(z_logit)
    return z_logit, z


@zs.reuse('variational')
def qh_net(x, z, n_h):
    with zs.BayesianNet() as variational:
        lh_x = layers.fully_connected(tf.to_float(tf.concat([x, z], axis=1)), 500)
        lh_x = layers.fully_connected(lh_x, 500)
        h_mean = layers.fully_connected(lh_x, n_h, activation_fn=None)
        h_logstd = layers.fully_connected(lh_x, n_h, activation_fn=None)
        h = zs.Normal('h', h_mean, logstd=h_logstd, group_ndims=1)
    return variational


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default='real', help='annotation_method')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    flgs, unparsed = parser.parse_known_args()

    # random seed
    np.random.seed(flgs.seed)
    tf.set_random_seed(flgs.seed)

    result_path = "results/mnist_scdc_{}_{}".format(
        time.strftime("%Y%m%d_%H%M%S"), flgs.seed)
    logger = setup_logger('mnist', __file__, result_path)

    # Load MNIST
    data_path = os.path.join('data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path, one_hot=False)

    x_train_orig = np.concatenate((x_train, x_valid), axis=0)
    t_train_orig = np.concatenate((t_train, t_valid), axis=0)

    binarize = lambda p: np.random.binomial(1, p, size=p.shape)
    x_test_bin = binarize(x_test)
    n_train, n_x = x_train_orig.shape  # input dimension

    # Define model parameters
    n_z = 50  # number of clusters
    n_h = 8  # dimension of latent space
    n = 128  # batch_size
    learning_rate = 0.001  # max learning rate

    # Define training parameters
    epoches = 200
    batch_size = n
    tt_bs = n
    iters = x_train_orig.shape[0] // batch_size
    test_iters = x_test.shape[0] // tt_bs
    save_freq = 1
    test_freq = 1

    # placeholders
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    lr = tf.placeholder(tf.float32, shape=[], name='lr')

    qz_logit, qz = qz_net(x, n_z)
    z_diag = tf.diag(tf.ones(n_z, dtype=tf.int32))
    z_u = tf.reshape(tf.tile(tf.expand_dims(z_diag, 0), [n, 1, 1]), [-1, n_z])
    x_u = tf.reshape(tf.tile(tf.expand_dims(x, 1), [1, n_z, 1]), [-1, n_x])
    variational = qh_net(x_u, z_u, n_h)
    qh_samples, log_qh = variational.query('h', outputs=True, local_log_prob=True)

    def log_joint(observed):
        n = tf.shape(observed['x'])[0]
        model, _, _ = vae(observed, n, n_x, n_h, n_z)
        log_px_h, log_ph_z, log_pz = model.local_log_prob(['x', 'h', 'z'])
        return log_px_h + log_ph_z + log_pz

    lb_z = zs.variational.elbo(
        log_joint, observed={'x': x_u, 'z': z_u},
        latent={'h': [qh_samples, log_qh]})
    lb_z = tf.reshape(lb_z, [-1, n_z])
    lower_bound = tf.reduce_mean(
        tf.reduce_sum(qz*lb_z, axis=1) +
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=qz, logits=qz_logit))
    z_pred = tf.argmax(qz, axis=1)

    ### relational part
    # generate annotations
    num_workers = 20
    num_annotated_points = 5000
    num_pairs_each_worker = 1000
    alpha = 0.9
    beta = 0.9
    if flgs.annotation == 'two_coin':
        rng = np.random.RandomState(100)
        annotations = two_coin(rng, alpha=alpha, beta=beta, num_worker=num_workers,
                               labels=t_train_orig[:num_annotated_points],
                               num_items_per_worker=num_pairs_each_worker,
                               full_adjacent=False, random_j_in_left=True)
    elif flgs.annotation == 'mygenConstraints':
        rng = np.random.RandomState(100)
        annotations = mygenConstraints(rng, t_train_orig[:num_annotated_points],
                                       alpha=[alpha] * num_workers,
                                       beta=[beta] * num_workers,
                                       num_ML=num_pairs_each_worker // 2,
                                       num_CL=num_pairs_each_worker // 2)
    elif flgs.annotation == 'genConstraints':
        rng = np.random.RandomState(100)
        annotations = genConstraints(rng, t_train_orig[:num_annotated_points],
                                     alpha=[alpha] * num_workers,
                                     beta=[beta] * num_workers,
                                     num_ML=num_pairs_each_worker // 2,
                                     num_CL=num_pairs_each_worker // 2,
                                     flag_same=True)
    elif flgs.annotation == 'real':  # load from real annotations for mnist
        path = 'data/crowd_annotations_mnist.npy'
        annotations = np.load(path)
        logger.info('Annotation shape: {}'.format(annotations.shape))
    else:
        raise NotImplementedError
    annotations = np.array(annotations)
    print(set(annotations[:, 2]))
    num_workers = len(set(annotations[:, 2]))
    logger.info('Worker number: {}'.format(num_workers))
    num_annotations = annotations.shape[0]
    ann_bs = annotations.shape[0] // iters
    coeff_rel = num_annotations / n_train
    logger.info('annotation batch size {}'.format(ann_bs))

    initial_alpha = np.random.beta(1, 1)
    initial_beta = np.random.beta(1, 1)

    prior_theta_worker = init_worker('worker_prior', num_workers, alpha=initial_alpha, beta=initial_beta)

    # placeholders
    x_i = tf.placeholder(tf.float32, shape=[None, n_x], name='xi')
    x_j = tf.placeholder(tf.float32, shape=[None, n_x], name='xj')
    batch_anns = tf.placeholder(tf.int32, shape=[None, 4],
                                name='batch_anns')
    qz_o_logits_i, qzi = qz_net(x_i, n_z)
    qz_o_logits_j, qzj = qz_net(x_j, n_z)

    # rel
    lb_rel = loglike_annotations(qz_o_logits_i, qz_o_logits_j,
                                 prior_theta_worker, batch_anns)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    infer = optimizer.minimize(-lower_bound - coeff_rel * lb_rel)

    # Generate images
    n_gen = 10*n_z
    z_gen = np.zeros([n_gen, n_z])
    for i in range(n_z):
        z_gen[i*10:(i+1)*10, i] = 1
    z_gen = tf.constant(z_gen, dtype=tf.int32)
    z_gen = tf.reshape(tf.transpose(tf.reshape(z_gen, [n_z, 10, n_z]), [1, 0, 2]), [-1, n_z])
    _, x_logits, pi = vae({'z': z_gen}, n_gen, n_x, n_h, n_z)
    transp = tf.exp(pi) / tf.reduce_max(tf.exp(pi))
    x_gen = tf.reshape(tf.sigmoid(x_logits), [10, n_z, 28, 28, 1]) * transp[..., None, None, None]
    x_gen = tf.reshape(x_gen, [-1, 28, 28, 1])

    def _evaluate(pred_batches, labels):
        preds = np.hstack(pred_batches)
        truths = labels[:preds.size]
        acc, _ = cluster_acc(preds, truths)
        nmi = adjusted_mutual_info_score(truths, labels_pred=preds)
        return acc, nmi

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epoches + 1):
            time_train = time.time()
            indices = np.random.permutation(x_train_orig.shape[0])
            x_train = x_train_orig[indices]
            t_train = t_train_orig[indices]
            lbs = []
            lrels = []
            y_preds = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size, :]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})

                ann_indices = np.random.randint(0, num_annotations, size=ann_bs)
                ann_batch = annotations[ann_indices]

                xi = x_train_orig[ann_batch[:, 0]]
                xj = x_train_orig[ann_batch[:, 1]]
                _, lb, lbr, y_pred = sess.run([infer, lower_bound, lb_rel, z_pred],
                                         feed_dict={x: x_batch_bin,
                                                    lr: learning_rate,
                                                    x_i: xi,
                                                    x_j: xj,
                                                    batch_anns: ann_batch
                                                    })
                lbs.append(lb)
                lrels.append(lbr)
                y_preds.append(y_pred)

            train_acc, train_nmi = _evaluate(y_preds, t_train)
            pred_alpha, pred_beta = sess.run(
                [tf.sigmoid(prior_theta_worker[0]),
                 tf.sigmoid(prior_theta_worker[1])])

            logger.info("alpha: {}, beta: {}".format(np.mean(pred_alpha), np.mean(pred_beta)))

            logger.info('Epoch {} {:.2f}s: Lower bound = {:.4f} Lrel = {:.4f}'
                  ' clustering accuracy = {:.2f}% nmi score = {:.4f}'.format(
                epoch, time.time() - time_train, np.mean(lbs), np.mean(lrels),
                train_acc*100, train_nmi))

            if epoch % test_freq == 0:
                time_test = -time.time()
                tt_lbs = []
                tt_preds = []
                for tt in range(test_iters):
                    test_x_batch = x_test_bin[tt * tt_bs: (tt + 1) * tt_bs]
                    tt_pred, tt_lb = sess.run([z_pred, lower_bound],
                                              feed_dict={x: test_x_batch})
                    tt_preds.append(tt_pred)
                    tt_lbs.append(tt_lb)
                test_acc, test_nmi = _evaluate(tt_preds, t_test)
                time_test += time.time()
                logger.info('>>> TEST EPOCH {} ({:.1f}s) lb {:2f} accuracy: {:.2f}% '
                      'nmi score = {:.4f}'.
                      format(epoch, time_test, np.mean(tt_lbs),  100.*test_acc,
                             test_nmi))

                if epoch == epoches:
                    with open('results/mnist_scdc.txt', "a") as myfile:
                        myfile.write("seed: %d train_acc: %f train_nmi: %f "
                                     "test_acc: %f test_nmi: %f" % (
                            flgs.seed, train_acc, train_nmi, test_acc, test_nmi))
                        myfile.write('\n')
                        myfile.close()

            if epoch % save_freq == 0:
                images = sess.run(x_gen)
                name = os.path.join(result_path, "epoch.{}.png".format(epoch))
                utils.save_image_collections(images, name, shape=(10, n_z))
