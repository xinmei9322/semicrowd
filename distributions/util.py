#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


symmetrize = lambda X: (X + tf.matrix_transpose(X)) / 2.
outer = lambda x, y: x[..., :, None] * y[..., None, :]
normalize = lambda x: x / tf.reduce_sum(x, axis=-1, keepdims=True)


def exp_family_kl(nat_param_diff, stats, log_z_diff=None):
    # nat_param_diff: [..., p]
    # stats: [..., p]
    # log_z_diff: [...]
    unnormalized_term = tf.reduce_sum(nat_param_diff * stats, axis=-1)
    if log_z_diff is None:
        return unnormalized_term
    else:
        return unnormalized_term - log_z_diff
