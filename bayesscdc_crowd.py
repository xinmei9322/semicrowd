#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BayesSCDC model for MNIST with noisy annotations.
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
from distributions import niw, catgorical, mvn, dirichlet, beta
from distributions import normalize, exp_family_kl
from generate_annotation import two_coin, genConstraints, mygenConstraints


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("seed", 1234, """Random seed.""")


def get_global_params(scope, d, K, W, alpha, niw_conc, tau, random_scale=None,
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

    def init_beta_param(tau_1, tau_2):
        # TODO: add randomness for variational
        return beta.standard_to_natural(tau_1, tau_2)

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
        # [W, 2]
        alpha_nat = tf.stack([init_beta_param(tau, 1) for _ in range(W)])
        alpha_params = tf.get_variable(
            "alpha_params", dtype=tf.float32, initializer=alpha_nat,
            trainable=trainable)
        # [W, 2]
        beta_nat = tf.stack([init_beta_param(tau, 1) for _ in range(W)])
        beta_params = tf.get_variable(
            "beta_params", dtype=tf.float32, initializer=beta_nat,
            trainable=trainable)
    return dir_params, niw_params, alpha_params, beta_params


def global_expected_stats(global_params, d):
    dir_params, niw_params, alpha_params, beta_params = global_params
    # [K]
    dir_stats = dirichlet.expected_stats(dir_params)
    # [K, d + d^2 + 2]
    niw_stats = niw.expected_stats(niw_params, d)
    # [W, 2]
    alpha_stats = beta.expected_stats(alpha_params)
    # [W, 2]
    beta_stats = beta.expected_stats(beta_params)
    return dir_stats, niw_stats, alpha_stats, beta_stats


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


def z_mean_field(global_stats, x_stats, prev_z_stats, nb_weights=None):
    # dir_stats: [K]
    dir_stats, niw_stats, alpha_stats, beta_stats = global_stats
    # x_stats: [M, d + d^2], niw_stats: [K, d + d^2 + 2]
    M = tf.shape(x_stats)[0]
    # x_stats_pad: [M, d + d^2 + 2]
    x_stats_pad = tf.concat([x_stats, tf.ones([M, 2])], axis=-1)
    # z_nat_param: [M, K]
    z_nat_param = dir_stats + tf.matmul(x_stats_pad, niw_stats,
                                        transpose_b=True)
    if nb_weights is not None:
        # nb_weights: [M, M] (sparse), prev_z_stats: [M, K]
        # rel_term: [M, K]
        # TODO: verify if there is 1/2.
        rel_term = tf.sparse_tensor_dense_matmul(nb_weights, prev_z_stats)
        # z_nat_param: [M, K]
        z_nat_param += rel_term
    # z_stats: [M, K]
    z_stats = catgorical.expected_stats(z_nat_param)
    return z_nat_param, z_stats


def annotation_log_likelihood(beta_stats, z_inner_stats, L, I, nb_weights):
    # z_inner_stats: [M, M], nb_weights: [M, M] (sparse)
    # term_1: [M, M] (sparse)
    term_1 = z_inner_stats * nb_weights
    # L: [M, M, W] (sparse), beta_stats: [W, 2]
    # term_2: [M, M, W] (sparse)
    term_2 = L * (beta_stats[:, 1] - beta_stats[:, 0])
    # I: [M, M, W] (sparse)
    # term_3: [M, M, W] (sparse)
    term_3 = I * beta_stats[:, 0]
    # ret: []
    return 0.5 * (tf.sparse_reduce_sum(term_1) +
                  tf.sparse_reduce_sum(term_2) +
                  tf.sparse_reduce_sum(term_3))


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
    prior_dir_param, prior_niw_param, prior_alpha_param, prior_beta_param = \
        prior_global_params

    def _kl_helper(log_partition, param, prior_param, stats):
        nat_diff = param - prior_param
        log_z_diff = log_partition(param) - log_partition(prior_param)
        return exp_family_kl(nat_diff, stats, log_z_diff=log_z_diff)

    # dir_param: [K], niw_param: [K, d + d^2 + 2]
    # alpha_params: [W, 2], beta_params: [W, 2]
    dir_param, niw_param, alpha_param, beta_param = global_params
    # dir_stats: [K], niw_stats: [K, d + d^2 + 2]
    # alpha_stats: [W, 2], beta_stats: [W, 2]
    dir_stats, niw_stats, alpha_stats, beta_stats = global_stats
    # dir_kl: []
    dir_kl = _kl_helper(dirichlet.log_partition, dir_param, prior_dir_param,
                        dir_stats)
    # niw_kl: [K]
    niw_kl = _kl_helper(lambda x: niw.log_partition(x, d), niw_param,
                        prior_niw_param, niw_stats)
    # alpha_kl, beta_kl: [W]
    alpha_kl = _kl_helper(beta.log_partition, alpha_param, prior_alpha_param,
                          alpha_stats)
    beta_kl = _kl_helper(beta.log_partition, beta_param, prior_beta_param,
                         beta_stats)
    return dir_kl + tf.reduce_sum(niw_kl, axis=0) + tf.reduce_sum(
        alpha_kl, axis=0) + tf.reduce_sum(beta_kl, axis=0)


def elbo(log_po_term, local_kl_z, local_kl_x, global_kl, N,
         ann_ll=None, ann_subsample_factor=None):
    # log_po_term: [M]
    # local_kl_z: [M], local_kl_x: [M]
    # global_kl: []
    obj = tf.reduce_mean(log_po_term - local_kl_z - local_kl_x) - global_kl / N
    # ann_ll: []
    if ann_ll is not None:
        obj += ann_ll * ann_subsample_factor / N
    # ret: []
    return obj


def variational_message_passing(
        prior_global_params, global_params, o, o_dim, d, K, N,
        L=None, I=None, n_ann=None, ann_batch_size=None, n_iters=100):
    global_stats = global_expected_stats(global_params, d)
    dir_stats, niw_stats, alpha_stats, beta_stats = global_stats
    M = tf.shape(o)[0]

    # Initialize z_stats
    z_stats = normalize(tf.random_uniform([M, K], 1e-8, maxval=1))

    # Encode
    # h: [M, d], J: [M, d]
    h, J = encoder(o, d)
    # J: [M, d * d]
    J = tf.reshape(tf.matrix_diag(J), [M, d * d])
    # x_obs_param: [M, d + d * d]
    x_obs_param = tf.concat([h, J], axis=-1)

    # Prepare relational info
    if L is not None:
        # I, L: [M, M, W] (sparse), alpha_stats: [W, 2]
        # nb_weights_per_worker: [M, M, W] (sparse)
        nb_weights_per_worker = tf.sparse_add(
            (alpha_stats[:, 1] - beta_stats[:, 0]) * I,
            (alpha_stats[:, 0] - alpha_stats[:, 1] +
             beta_stats[:, 0] - beta_stats[:, 1]) * L)
        # nb_weights: [M, M] (sparse)
        nb_weights = tf.sparse_reduce_sum_sparse(nb_weights_per_worker, axis=-1)
    else:
        nb_weights = None

    # Message passing
    for t in range(n_iters):
        x_nat_param, x_stats = x_mean_field(niw_stats, z_stats, x_obs_param, d)
        z_nat_param, z_stats = z_mean_field(global_stats, x_stats, z_stats,
                                            nb_weights=nb_weights)

    # Decode
    # x: [M, d]
    x = mvn.sample(x_nat_param, d)
    o_dist, _ = decoder(x, o_dim)

    # Compute ELBO
    # log_po_term: [M]
    log_po_term = o_dist.log_prob(o)
    # log_p_ann_term: []
    if L is not None:
        # z_stats: [M, K], z_inner_stats: [M, M]
        z_inner_stats = tf.matmul(z_stats, z_stats, transpose_b=True)
        log_p_ann_term = annotation_log_likelihood(
            beta_stats, z_inner_stats, L, I, nb_weights)
        ann_subsample_factor = n_ann / ann_batch_size
    else:
        z_inner_stats = None
        log_p_ann_term = None
        ann_subsample_factor = 1
    # log_kl_x_term: [M]
    local_kl_x_term = local_kl_x(x_nat_param, niw_stats, z_stats, x_stats, d)
    # log_kl_z_term: [M]
    local_kl_z_term = local_kl_z(z_nat_param, dir_stats, z_stats)
    # global_kl_term: []
    global_kl_term = global_kl(
        prior_global_params, global_params, global_stats, d)
    lower_bound = elbo(
        log_po_term, local_kl_z_term, local_kl_x_term, global_kl_term, N,
        ann_ll=log_p_ann_term, ann_subsample_factor=ann_subsample_factor)

    # Natural gradient for global variational parameters
    # z_stats: [M, K], x_stats: [M, d + d^2]
    # dir_updates: [K]
    dir_updates = tf.reduce_mean(z_stats, axis=0)
    # niw_updates: [K, d + d^2 + 2]
    niw_updates = tf.matmul(z_stats, tf.concat([x_stats, tf.ones([M, 2])], -1),
                            transpose_a=True) / tf.to_float(M)
    updates = [dir_updates, niw_updates]

    if L is not None:
        # L_worker: [W, M, M] (sparse), false_L_worker: [W, M, M] (sparse)
        L_worker = tf.sparse_transpose(L, perm=[2, 0, 1])
        false_L_worker = tf.sparse_transpose(
            tf.sparse_add(I, -tf.ones(tf.shape(L)) * L), perm=[2, 0, 1])
        # alpha_updates: [W, 2]
        alpha_updates_1 = tf.sparse_reduce_sum(z_inner_stats * L_worker,
                                               axis=(-2, -1))
        alpha_updates_2 = tf.sparse_reduce_sum(z_inner_stats * false_L_worker,
                                               axis=(-2, -1))
        alpha_updates = 0.5 * tf.stack([alpha_updates_1, alpha_updates_2],
                                       axis=-1)
        # beta_updates: [W, 2]
        # false_z_inner_stats: [M, M]
        false_z_inner_stats = 1 - z_inner_stats
        beta_updates_1 = tf.sparse_reduce_sum(
            false_z_inner_stats * false_L_worker, axis=(-2, -1))
        beta_updates_2 = tf.sparse_reduce_sum(
            false_z_inner_stats * L_worker, axis=(-2, -1))
        beta_updates = 0.5 * tf.stack([beta_updates_1, beta_updates_2], axis=-1)
        updates.extend([alpha_updates / ann_subsample_factor,
                        beta_updates / ann_subsample_factor])

    nat_grads = [(prior_global_params[i] - global_params[i]) / N + updates[i]
                 for i in range(len(updates))]
    return lower_bound, nat_grads, z_stats, niw_stats, dir_stats


def load_annotations(t_train, W, method="two_coin"):
    num_workers = W
    num_annotated_points = 5000
    num_pairs_each_worker = 1000
    if method == 'two_coin':
        rng = np.random.RandomState(100)
        annotations = two_coin(rng, alpha=0.9, beta=0.9, num_worker=num_workers,
                               labels=t_train[:num_annotated_points],
                               num_items_per_worker=num_pairs_each_worker,
                               random_j_in_left=True,
                               full_adjacent=False)
    elif method == 'mygenConstraints':
        rng = np.random.RandomState(100)
        annotations = mygenConstraints(rng, t_train[:num_annotated_points],
                                       alpha=[0.9] * num_workers,
                                       beta=[0.9] * num_workers,
                                       num_ML=num_pairs_each_worker // 2,
                                       num_CL=num_pairs_each_worker // 2)
    elif method == 'genConstraints':
        rng = np.random.RandomState(100)
        annotations = genConstraints(rng, t_train[:num_annotated_points],
                                     alpha=[0.9] * num_workers,
                                     beta=[0.9] * num_workers,
                                     num_ML=num_pairs_each_worker // 2,
                                     num_CL=num_pairs_each_worker // 2,
                                     flag_same=True)
    elif method == 'real':  # load from real annotations for mnist
        path = 'data/crowd_annotations_mnist.npy'
        annotations = np.load(path)
        print('Annotation shape: {}'.format(annotations.shape))
    else:
        raise NotImplementedError
    annotations = np.array(annotations)
    return annotations


def make_sparse_ann_batch(ann_batch, W):
    orig_indices = []
    for i, j, w, l in ann_batch:
        orig_indices.append(i)
        orig_indices.append(j)
    orig_indices = sorted(list(set(orig_indices)))

    batch_ind_to_orig = {}
    orig_to_batch_ind = {}
    for i, ind in enumerate(orig_indices):
        orig_to_batch_ind[ind] = i
        batch_ind_to_orig[i] = ind

    indices = []
    values = []
    for i, j, w, l in ann_batch:
        indices.append([orig_to_batch_ind[i], orig_to_batch_ind[j], w])
        indices.append([orig_to_batch_ind[j], orig_to_batch_ind[i], w])
        values.append(l)
        values.append(l)
    shape = [len(orig_indices), len(orig_indices), W]
    sparse_ann_batch = tf.SparseTensorValue(indices, values, shape)
    sparse_ind_batch = tf.SparseTensorValue(
        indices, np.ones_like(values), shape)
    return (orig_indices, orig_to_batch_ind, batch_ind_to_orig,
            sparse_ann_batch, sparse_ind_batch)


def main():
    seed = FLAGS.seed
    result_path = "results/mnist_crowd_{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), seed)
    logger = setup_logger('mnist', __file__, result_path)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Load MNIST
    data_path = os.path.join('data', 'mnist.pkl.gz')
    o_train, t_train, o_valid, t_valid, o_test, t_test = \
        dataset.load_mnist_realval(data_path, one_hot=False)
    o_train = np.vstack([o_train, o_valid])
    t_train = np.hstack([t_train, t_valid])
    n_train, o_dim = o_train.shape
    # indices = np.random.permutation(n_train)
    # o_train = o_train[indices]
    # t_train = t_train[indices]
    o_test = np.random.binomial(1, o_test, size=o_test.shape)
    n_test, _ = o_test.shape
    # n_class = np.max(t_test) + 1

    # Prior parameters
    d = 8
    K = 50
    W = 20
    prior_alpha = 1.05
    prior_niw_conc = 0.5
    prior_tau = 1.

    # Variational initialization
    alpha = 2.
    niw_conc = 1.
    random_scale = 3.
    tau = 10.

    # learning rate
    learning_rate = 1e-3
    nat_grad_scale = 1e4

    # Load annotations
    # [i, j, w, L]
    annotations = load_annotations(t_train, W, method="real")
    n_annotations = annotations.shape[0]
    W = len(set(annotations[:, 2]))
    # batch_size = 128
    # iters = o_train.shape[0] // batch_size
    # ann_batch_size = annotations.shape[0] // iters
    # print(ann_batch_size)
    # exit(0)

    # Define training parameters
    epochs = 200
    batch_size = 128
    iters = o_train.shape[0] // batch_size
    ann_batch_size = annotations.shape[0] // iters
    save_freq = 1
    test_freq = 10
    test_batch_size = 400
    test_iters = o_test.shape[0] // test_batch_size

    prior_global_params = get_global_params(
        "prior", d, K, W, prior_alpha, prior_niw_conc, prior_tau,
        trainable=False)
    global_params = get_global_params(
        "variational", d, K, W, alpha, niw_conc, tau,
        random_scale=random_scale, trainable=True)

    # n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    o_input = tf.placeholder(tf.float32, shape=[None, o_dim], name='o')
    o = tf.to_int32(tf.random_uniform(tf.shape(o_input)) <= o_input)

    ann_o_input = tf.placeholder(tf.float32, shape=[None, o_dim], name='ann_o')
    ann_o = tf.to_int32(tf.random_uniform(tf.shape(ann_o_input)) <= ann_o_input)
    L_ph = tf.sparse_placeholder(tf.float32, shape=[None, None, W])
    I_ph = tf.sparse_placeholder(tf.float32, shape=[None, None, W])

    lower_bound, global_nat_grads, z_stats, niw_stats, dir_stats = \
        variational_message_passing(
            prior_global_params, global_params, o, o_dim, d, K, n_train,
            n_iters=4)
    z_pred = tf.argmax(z_stats, axis=-1)

    ann_lower_bound, ann_nat_grads, _, _, _ = variational_message_passing(
        prior_global_params, global_params, ann_o, o_dim, d, K, n_train,
        L_ph, I_ph, n_annotations, ann_batch_size, n_iters=4)
    # ann_lower_bound = tf.constant(0.)
    # ann_nat_grads = [tf.zeros_like(param) for param in global_params]

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9)
    net_vars = (tf.trainable_variables(scope="encoder") +
                tf.trainable_variables(scope="decoder"))
    net_grads_and_vars = optimizer.compute_gradients(
        -0.5 * (lower_bound + ann_lower_bound), var_list=net_vars)
    global_nat_grads.extend([0, 0])
    nat_grads = [-nat_grad_scale * 0.5 * (g + ann_g)
                 for g, ann_g in zip(global_nat_grads, ann_nat_grads)]
    global_grads_and_vars = list(zip(nat_grads, global_params))
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
            # print(indices[:5])
            # exit(0)
            o_train_raw = o_train[indices]
            t_train_raw = t_train[indices]
            lbs, ann_lbs = [], []
            t_preds, ann_t_preds = [], []
            for t in range(iters):
                # Without annotation
                o_batch = o_train_raw[t * batch_size:(t + 1) * batch_size]

                # With annotation
                ann_indices = np.random.randint(0, n_annotations,
                                                size=ann_batch_size)
                ann_batch = annotations[ann_indices]
                o_indices, orig_to_batch_ind, batch_to_orig_ind, \
                    sparse_ann_batch, sparse_ann_ind = make_sparse_ann_batch(
                        ann_batch, W)
                ann_o_batch = o_train[o_indices]

                _, lb, t_pred, ann_lb = sess.run(
                    [infer_op, lower_bound, z_pred, ann_lower_bound],
                    feed_dict={o_input: o_batch,
                               ann_o_input: ann_o_batch,
                               L_ph: sparse_ann_batch,
                               I_ph: sparse_ann_ind})
                lbs.append(lb)
                t_preds.append(t_pred)
                # print("lb: {}".format(lb))
                ann_lbs.append(ann_lb)

            time_epoch += time.time()
            train_acc, train_nmi = _evaluate(t_preds, t_train_raw)
            logger.info(
                'Epoch {} ({:.1f}s): Lower bound = {}, ann LB = {}, '
                'acc = {}, nmi = {}'
                .format(epoch, time_epoch, np.mean(lbs), np.mean(ann_lbs),
                        train_acc, train_nmi))

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
                    with open('results/mnist_bayesSCDC.txt', "a") as myfile:
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
