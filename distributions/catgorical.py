#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def expected_stats(nat_param):
    # nat_param: [..., K]
    # ret: [..., K]
    return tf.nn.softmax(nat_param)
