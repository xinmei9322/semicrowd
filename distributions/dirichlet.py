#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def expected_stats(nat_param):
    # nat_param: [..., K]
    alpha = nat_param + 1
    # ret: [..., K]
    return tf.digamma(alpha) - tf.digamma(
        tf.reduce_sum(alpha, -1, keepdims=True))


def log_partition(nat_param):
    # nat_param: [..., K]
    alpha = nat_param + 1
    # ret: [...]
    return tf.reduce_sum(tf.lgamma(alpha), axis=-1) - tf.lgamma(
        tf.reduce_sum(alpha, axis=-1))
