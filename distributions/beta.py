#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def expected_stats(nat_param):
    # nat_param: [..., 2]
    tau = nat_param + 1
    # ret: [..., 2]
    return tf.digamma(tau) - tf.digamma(
        tf.reduce_sum(tau, axis=-1, keepdims=True))


def log_partition(nat_param):
    # nat_param: [..., 2]
    tau = nat_param + 1
    # ret: [...]
    return tf.reduce_sum(tf.lgamma(tau), axis=-1) - tf.lgamma(
        tf.reduce_sum(tau, axis=-1))


def standard_to_natural(tau_1, tau_2):
    # tau_1, tau_2: [...]
    return tau_1 - 1, tau_2 - 1


def natural_to_standard(nat_param):
    # nat_param: [..., 2]
    return nat_param[..., 0] + 1, nat_param[..., 1] + 1
