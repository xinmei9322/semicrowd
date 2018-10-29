#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from .util import outer, symmetrize


def pack_struct4(b, A, c, d):
    # A: b: [..., p], [..., p, p], c: [..., 1], d: [..., 1]
    # vec_A: [..., p * p]
    vec_A = tf.reshape(A, tf.concat([tf.shape(A)[:-2], [-1]], 0))
    # ret: [..., p + p^2 + 2]
    return tf.concat([b, vec_A, c, d], axis=-1)


def unpack_struct4(struct, p):
    # struct: [..., p + p^2 + 2]
    b, vec_A, c, d = tf.split(struct, [p, p * p, 1, 1], axis=-1)
    # A: [..., p, p]
    A = tf.reshape(vec_A, tf.concat([tf.shape(vec_A)[:-1], [p, p]], 0))
    # ret: b: [..., p], A: [..., p, p], c: [..., 1], d: [..., 1]
    return b, A, c, d


def multi_digamma(x, p):
    # x: [..., 1]
    # gamma_seq: [p]
    gamma_seq = tf.range(p, dtype=tf.float32)
    # ret: [..., 1]
    return tf.reduce_sum(tf.digamma(x - gamma_seq / 2), axis=-1, keepdims=True)


def log_multi_gamma(x, p):
    # x: [..., 1]
    # gamma_seq: [p]
    gamma_seq = tf.range(p, dtype=tf.float32)
    # ret: [...]
    return 0.25 * p * (p - 1) * tf.log(np.pi) + tf.reduce_sum(
        tf.lgamma(x - gamma_seq / 2), axis=-1)


def standard_to_natural(m, kappa, S, nu):
    # m: [..., p]
    # kappa: [...]
    # S: [..., p, p]
    # nu: [...]
    # kappa: [..., 1]
    kappa = kappa[..., None]
    # b: [..., p]
    b = kappa * m
    # ret: [..., p + p^2 + 2]
    return pack_struct4(
        kappa * m,
        S + outer(b, m),
        kappa,
        nu[..., None] + tf.to_float(tf.shape(m)[-1]) + 2)


def natural_to_standard(nat_param, p):
    b, A, c, d = unpack_struct4(nat_param, p)
    # kappa: [..., 1]
    kappa = c
    # nu: [..., 1]
    nu = d - p - 2
    # m: [..., p]
    m = b / kappa
    # S: [..., p, p]
    S = A - outer(b, m)
    return m, kappa, S, nu


def expected_stats(nat_param, p, eps=1e-5):
    # nat_param: [..., p + p^2 + 2]
    # m: [..., p], kappa: [..., 1], S: [..., p, p], nu: [..., 1]
    m, kappa, S, nu = natural_to_standard(nat_param, p)

    # S_inv = symmetrize(tf.matrix_inverse(S + eps * tf.eye(p)))
    S_inv = symmetrize(tf.matrix_inverse(S))
    # S_inv_m: [..., p]
    S_inv_m = tf.squeeze(tf.matmul(S_inv, m[..., None]), -1)
    # stats_b: [..., p]
    stats_b = nu * S_inv_m
    # stats_A: [..., p, p]
    stats_A = -0.5 * nu[..., None] * S_inv
    # stats_c: [..., 1]
    stats_c = -0.5 * (p / kappa +
                      tf.reduce_sum(stats_b * m, axis=-1, keepdims=True))
    # stats_d: [..., 1]
    stats_d = 0.5 * (multi_digamma(0.5 * nu, p) + p * tf.log(2.) -
                     tf.linalg.logdet(symmetrize(S))[..., None])
    # ret: [..., p + p^2 + 2]
    return pack_struct4(stats_b, stats_A, stats_c, stats_d)


def log_partition(nat_param, p):
    # m: [..., p], kappa: [..., 1], S: [..., p, p], nu: [..., 1]
    m, kappa, S, nu = natural_to_standard(nat_param, p)
    # kappa: [...]
    kappa = tf.squeeze(kappa, -1)
    S = symmetrize(S)
    # ret: [...]
    return 0.5 * tf.squeeze(nu, -1) * (p * tf.log(2.) - tf.linalg.logdet(S)) + \
        log_multi_gamma(0.5 * nu, p) - 0.5 * p * tf.log(kappa)
