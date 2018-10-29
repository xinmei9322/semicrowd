#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import scipy.io as sio
import tensorflow as tf
from six.moves import range

from utils import dataset

# I/O
# tf.flags.DEFINE_string('save_dir',
#                        os.environ['MODEL_RESULT_PATH_AND_PREFIX'],
#                        'Location for parameter checkpoints and samples')
flgs = tf.flags.FLAGS  # Define training/evaluation parameters


def group_by_class(x, t):
    x_by_class = []
    t_by_class = []
    indices_by_class = []
    for i in range(10):
        indices = np.nonzero(t == i)[0]
        x_by_class.append(x[indices])
        t_by_class.append(t[indices])
        indices_by_class.append(indices)
    return x_by_class, t_by_class, indices_by_class


def select_by_class(x_by_class, t_by_class, indices_by_class, n_l):
    # rng = np.random.RandomState(seed=seed)
    x_labeled = []
    t_labeled = []
    indices_labeled = []
    for i in range(10):
        indices = np.arange(x_by_class[i].shape[0])
        # rng.shuffle(indices)
        np.random.shuffle(indices)
        x_labeled.append(x_by_class[i][indices[:n_l]])
        t_labeled.append(t_by_class[i][indices[:n_l]])
        indices_labeled.append(indices_by_class[i][indices[:n_l]])
    x_labeled = np.vstack(x_labeled)
    t_labeled = np.hstack(t_labeled)
    indices_labeled = np.hstack(indices_labeled)
    return x_labeled, t_labeled, indices_labeled


def worker(t, ind, corruption_percentage, num_classes, worker_id):
    # annotations = np.zeros(int(0.5*len(t)*(len(t)-1)))
    # label_id = np.zeros(3, len(annotations))
    annotations = []
    label_id = []
    indices = np.arange(len(t))
    np.random.shuffle(indices)
    t = t[indices]
    ind = ind[indices]
    if corruption_percentage > 0:
        corrupt_labels = int(len(t) * corruption_percentage)
        for i in range(corrupt_labels):
            t[i] = np.random.randint(0, num_classes)
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            label_id.append([ind[i], ind[j], worker_id])
            if t[i] == t[j]:
                annotations.append(1)
            else:
                annotations.append(-1)
    return annotations, label_id


def two_coin(rng, alpha, beta, num_worker, labels, num_items_per_worker, full_adjacent=False, random_j_in_left=False):
    """

    :param alpha: sensitivity, a list with length num_worker or a scalar in (0.5, 1].
    :param beta:  specificity, a list with length num_worker or a scalar in (0.5, 1].
    :param num_worker: int, a non-negative scalar, should be equal to len(alpha)
        if alpha is a list
    :param labels: The true labels of the samples.
    :param num_items_per_worker: int. Same for all workers
    :param full_adjacent: if True, generate n(n-1)/2 pairs
    :param random_j_in_left: if True, choose j in the left from i+1 to the end.
    :return:
    """
    if isinstance(alpha, list):
        assert len(alpha) == num_worker
    if isinstance(beta, list):
        assert len(beta) == num_worker
    anns = []
    ids = []
    mask_id = np.zeros(len(labels))
    end_id = num_items_per_worker
    indices = np.arange(len(labels))  # indices of the whole dataset

    def two_coin_each_worker(t, ind, worker_id, alpha_worker, beta_worker):
        annotations = []
        label_id = []
        indices_batch = np.arange(len(t))
        rng.shuffle(indices_batch)
        t = t[indices_batch]
        ind = ind[indices_batch]
        for i in range(len(t) - 1):
            if full_adjacent:
                set_j = range(i + 1, len(t))
                for j in set_j:
                    label_id.append([ind[i], ind[j], worker_id])
                    if t[i] == t[j]:
                        label = 1 if rng.rand() < alpha_worker else 0
                        annotations.append(label)
                    else:
                        label = 0 if rng.rand() < beta_worker else 1
                        annotations.append(label)
            else:
                if random_j_in_left:
                    j = rng.randint(i+1, len(t), 1)[0]
                else:
                    j = rng.randint(0, len(t), 1)[0]
                    while j == i:
                        j = rng.randint(0, len(t), 1)[0]
                if t[i] == t[j]:
                    label = 1 if rng.rand() < alpha_worker else 0
                    annotations.append(label)
                else:
                    label = 0 if rng.rand() < beta_worker else 1
                    annotations.append(label)
                label_id.append([ind[i], ind[j], worker_id])

        return annotations, label_id

    for j in range(num_worker):
        indices_labeled = indices[end_id - num_items_per_worker:end_id]  # indices of data labeled by worker j
        t_labeled = labels[end_id-num_items_per_worker:end_id]
        alpha_j = alpha[j] if isinstance(alpha, list) else alpha
        beta_j = beta[j] if isinstance(beta, list) else beta
        ann, id = two_coin_each_worker(t_labeled, indices_labeled, j, alpha_j, beta_j)
        anns.append(ann)
        ids.append(id)
        for ll in indices_labeled:
            mask_id[ll] += 1
        end_id += num_items_per_worker
        if end_id > len(labels):
            shuffle = rng.permutation(len(indices))
            indices = indices[shuffle]
            labels = labels[shuffle]
            end_id = num_items_per_worker

    anns = np.hstack(anns)
    ids = np.vstack(ids)
    id_anns = np.hstack((ids, anns[:, None]))
    print('The shape of annotations is', id_anns.shape)
    return id_anns


def mygenConstraints(prng, label, alpha, beta, num_ML, num_CL):
    """ This function generates pairwise constraints (ML/CL) using groud-truth
    cluster label and noise parameters
    Parameters
    ----------
    label: shape(n_sample, )
        cluster label of all the samples
    alpha: shape(n_expert, )
        sensitivity parameters of experts
    beta: shape(n_expert, )
        specificity parameters of experts
    num_ML: int
    num_CL: int

    Returns
    -------
    S: shape(n_con, 4)
        The first column -> expert id
        The second and third column -> (row, column) indices of two samples
        The fourth column -> constraint values (1 for ML and 0 for CL)
    """
    start_time = time.time()
    n_sample = len(label)
    # get indices of upper-triangle matrix
    [row, col] = np.triu_indices(n_sample, k=1)

    # generate noisy constraints for each expert
    assert len(alpha) == len(beta)
    n_expert = len(alpha)
    n_pairs = len(row)

    # initialize the constraint matrix
    S = np.zeros((0, 4))

    # different experts provide constraints for different sets of sample pairs
    for m in range(n_expert):
        ml_set = []
        cl_set = []
        while len(ml_set) < num_ML or len(cl_set) < num_CL:
            idx = prng.randint(0, n_pairs, 1)[0]
            if label[row[idx]] == label[col[idx]]:
                if len(ml_set) < num_ML:
                    ml_set.append([row[idx], col[idx]])
            elif label[row[idx]] != label[col[idx]]:
                if len(cl_set) < num_CL:
                    cl_set.append([row[idx], col[idx]])
            else:
                print("Invalid matrix entry values")
        ml_set = np.array(ml_set)
        cl_set = np.array(cl_set)

        val_ml = prng.binomial(1, alpha[m], num_ML)
        val_cl = prng.binomial(1, 1 - beta[m], num_CL)
        sm_ml = np.hstack((ml_set, np.ones((num_ML, 1)) * m,
                           val_ml.reshape(val_ml.size, 1)))
        sm_cl = np.hstack((cl_set, np.ones((num_CL, 1)) * m,
                           val_cl.reshape(val_cl.size, 1)))
        S = np.vstack((S, sm_ml, sm_cl)).astype(int)
    print('Generating annotations time: {:.2f}s'.format(time.time() - start_time))
    return S


def genConstraints(prng, label, alpha, beta, num_ML, num_CL, start_expert=0,
                   flag_same=False):
    """ This function generates pairwise constraints (ML/CL) using groud-truth
    cluster label and noise parameters
    Parameters
    ----------
    label: shape(n_sample, )
        cluster label of all the samples
    alpha: shape(n_expert, )
        sensitivity parameters of experts
    beta: shape(n_expert, )
        specificity parameters of experts
    num_ML: int
    num_CL: int
    flag_same: True if different experts provide constraints for the same set
    of sample pairs, False if different experts provide constraints for
    different set of sample pairs

    Returns
    -------
    S: shape(n_con, 4)
        The first column -> expert id
        The second and third column -> (row, column) indices of two samples
        The fourth column -> constraint values (1 for ML and 0 for CL)
    """
    n_sample = len(label)
    tp = np.tile(label, (n_sample, 1))
    # label_mat = (tp == tp.T).astype(int)

    ML_set = []
    CL_set = []
    # get indices of upper-triangle matrix
    [row, col] = np.triu_indices(n_sample, k=1)
    # n_sample * (n_sample-1)/2
    for idx in range(len(row)):
        if label[row[idx]] == label[col[idx]]:
            ML_set.append([row[idx], col[idx]])
        elif label[row[idx]] != label[col[idx]]:
            CL_set.append([row[idx], col[idx]])
        # if label_mat[row[idx], col[idx]] == 1:
        #     ML_set.append([row[idx], col[idx]])
        # elif label_mat[row[idx], col[idx]] == 0:
        #     CL_set.append([row[idx], col[idx]])
        else:
            print("Invalid matrix entry values")

    ML_set = np.array(ML_set)
    CL_set = np.array(CL_set)

    assert num_ML < ML_set.shape[0]
    assert num_CL < CL_set.shape[0]

    # generate noisy constraints for each expert
    assert len(alpha) == len(beta)
    n_expert = len(alpha)

    # initialize the constraint matrix
    S = np.zeros((0, 4))

    # different experts provide constraint for the same set of sample pairs
    if flag_same == True:
        idx_ML = prng.choice(ML_set.shape[0], num_ML, replace=False)
        idx_CL = prng.choice(CL_set.shape[0], num_CL, replace=False)
        ML = ML_set[idx_ML, :]
        CL = CL_set[idx_CL, :]
        for m in range(n_expert):
            val_ML = prng.binomial(1, alpha[m], num_ML)
            val_CL = prng.binomial(1, 1 - beta[m], num_CL)
            Sm_ML = np.hstack((ML, np.ones((num_ML, 1)) * (m + start_expert),
                               val_ML.reshape(val_ML.size, 1)))
            Sm_CL = np.hstack((CL, np.ones((num_CL, 1)) * (m + start_expert),
                               val_CL.reshape(val_CL.size, 1)))
            S = np.vstack((S, Sm_ML, Sm_CL)).astype(int)
    # different experts provide constraints for different sets of sample pairs
    else:
        for m in range(n_expert):
            prng = np.random.RandomState(1000 + m)
            idx_ML = prng.choice(ML_set.shape[0], num_ML, replace=False)
            idx_CL = prng.choice(CL_set.shape[0], num_CL, replace=False)
            ML = ML_set[idx_ML, :]
            CL = CL_set[idx_CL, :]
            val_ML = prng.binomial(1, alpha[m], num_ML)
            val_CL = prng.binomial(1, 1 - beta[m], num_CL)
            Sm_ML = np.hstack((ML, np.ones((num_ML, 1)) * (m + start_expert),
                               val_ML.reshape(val_ML.size, 1)))
            Sm_CL = np.hstack((CL, np.ones((num_CL, 1)) * (m + start_expert),
                               val_CL.reshape(val_CL.size, 1)))
            S = np.vstack((S, Sm_ML, Sm_CL)).astype(int)

    return S


if __name__ == "__main__":
    # Load MNIST
    data_path = os.path.join(
        'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path, one_hot=False)
    n_x = x_train.shape[1]
    n_y = 10
    n_j = 400  # number of workers
    n_l = 20  # number of items each worker annotates each class
    n_l_all = 100  # number of items each worker annotates
    confusion_rate = 0
    # x_by_class, t_by_class, ind_by_class = group_by_class(x_test, t_test)
    anns = []
    ids = []
    mask_id = np.zeros(len(t_test))  # whether the id is labeled
    end_id = n_l_all
    indices = np.arange(len(t_test))
    for j in range(n_j):
        # _, t_labeled, indices_labeled = select_by_class(x_by_class, t_by_class, ind_by_class, n_l)
        indices_labeled = indices[end_id-n_l_all:end_id]
        t_labeled = t_test[end_id-n_l_all:end_id]
        ann, id = worker(t_labeled, indices_labeled, confusion_rate, n_y, j)
        anns.append(ann)
        ids.append(id)
        for ll in indices_labeled:
            mask_id[ll] += 1
        end_id += n_l_all
        if end_id > len(t_test):
            shuffle = np.random.permutation(len(indices))
            indices = indices[shuffle]
            t_test = t_test[shuffle]
            end_id = n_l_all

    # plt.hist(mask_id)
    # plt.show()
    # plt.close()

    anns = np.hstack(anns)
    ids = np.vstack(ids).transpose()
    ids = ids + 1
    sio.savemat('L_and_label_id.mat', {'L': anns, 'label_id': ids})
    print('The length of annotations is', len(anns), ids.shape)
    print(anns[:200])






