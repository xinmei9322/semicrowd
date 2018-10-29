#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from .util import outer, symmetrize


def pack_struct2(h, J):
    # h: [..., p], J: [..., p, p]
    # vec_J: [..., p * p]
    vec_J = tf.reshape(J, tf.concat([tf.shape(J)[:-2], [-1]], 0))
    # ret: [..., p + p^2]
    return tf.concat([h, vec_J], axis=-1)


def unpack_struct2(struct, p):
    # struct: [..., p + p^2]
    # h: [..., p], vec_J: [..., p * p]
    h, vec_J = tf.split(struct, [p, p * p], axis=-1)
    # J: [..., p, p]
    J = tf.reshape(vec_J, tf.concat([tf.shape(vec_J)[:-1], [p, p]], 0))
    return h, J


def natural_to_standard(nat_param, p, eps=1e-5):
    # nat_param: [..., p + p^2]
    # h: [..., p], # J: [..., p, p]
    h, J = unpack_struct2(nat_param, p)
    # sigma_inv: [..., p, p]
    sigma_inv = -2 * J
    # sigma_inv += eps * tf.eye(p)
    # L: [..., p, p]
    # TODO: check whether matrix_solve or cholesky_solve.
    L = tf.cholesky(sigma_inv)
    # mu: [..., p, 1]
    mu = tf.cholesky_solve(L, h[..., None])
    return sigma_inv, L, mu


def sample(nat_param, p, n_samples=None):
    # nat_param: [..., p + p^2]
    sigma_inv, L, mu = natural_to_standard(nat_param, p)
    sample_dim = [1] if n_samples is None else [n_samples]
    sample_shape = tf.concat([tf.shape(sigma_inv)[:-1], sample_dim], 0)
    # sigma^1/2 noise: [..., p, n_samples]
    half_sigma_noise = tf.matrix_solve(L, tf.random_normal(sample_shape),
                                       adjoint=True)
    # ret: [..., p, n_samples]
    ret = mu + half_sigma_noise
    if n_samples is None:
        ret = tf.squeeze(ret, axis=-1)
    return ret


def expected_stats(nat_param, p):
    # nat_param: [..., p + p^2]
    sigma_inv, L, mu = natural_to_standard(nat_param, p)
    # mu: [..., p]
    mu = tf.squeeze(mu, -1)
    # sigma: [..., p, p]
    sigma = tf.matrix_inverse(sigma_inv)
    sigma = symmetrize(sigma)
    # stats2: [..., p, p]
    stats2 = sigma + outer(mu, mu)
    return pack_struct2(mu, stats2)


def log_partition(nat_param, p):
    # nat_param: [..., p + p^2]
    sigma_inv, L, mu = natural_to_standard(nat_param, p)
    # sigma_inv_mu: [..., p]
    sigma_inv_mu = tf.squeeze(tf.matmul(sigma_inv, mu), -1)
    # mu: [..., p]
    mu = tf.squeeze(mu, -1)
    # ret: [...]
    return 0.5 * (tf.reduce_sum(mu * sigma_inv_mu, axis=-1) -
                  tf.linalg.logdet(sigma_inv))
