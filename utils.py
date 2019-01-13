# -*- coding: utf-8 -*-
"""
   File Name：     utils
   Description :
   Author :       mick.yi
   date：          2019/1/4
"""
import numpy as np


def enqueue(np_array, elem):
    """
    入队列，新增元素放到队首，队尾元素丢弃
    :param np_array: 原始队列
    :param elem: 增加元素
    :return:
    """
    np_array[1:] = np_array[:-1]
    np_array[0] = elem
    return np_array


def random_select(ids):
    """
    随机选择一个id
    :param ids: id列表,(N,)
    :return:
    """
    idx = np.random.choice(len(ids))
    return ids[idx]


# def to_train_label(train_label):


def update_weights(h5_file, h5_dataset, weights, labels):
    """
    更新保存在hdf5中的原型权重
    :param h5_file: 原型权重的hdf5文件
    :param h5_dataset:  原型权重在hdf5中的dataset
    :param weights: 待更新的权重，numpy数组 (Batch,Dim)
    :param labels: 待更新的权重对应的类别标签
    :return:
    备注：TypeError: Indexing elements must be in increasing order; idx要排序
    TypeError: PointSelection __getitem__ only works with bool arrays; labels[idx]改为list(labels[idx])
    """
    # for idx, label in enumerate(labels):
    #     h5_dataset[label] = weights[idx]
    idx = np.argsort(labels)
    h5_dataset[list(labels[idx])] = weights[idx]
    h5_file.flush()


def get_weights(h5_dataset, labels):
    weights = [h5_dataset[label] for label in labels]
    return np.asarray(weights)


def update_queue(dominant_queue, candidate_queue, predict, current_labels):
    """
    更新支配队列
    :param dominant_queue: 支配队列
    :param candidate_queue: 候选队列
    :param predict: 预测的类别，numpy数组 (Batch,Batch)
    :param current_labels: 实际的当前类别
    :return:
    """
    predict_label = np.argmax(predict, axis=-1)
    for i in range(len(predict_label)):
        d_label_queue = dominant_queue[current_labels[i]]
        c_label_queue = candidate_queue[current_labels[i]]
        real_predict_label = current_labels[predict_label[i]]
        # 预测结果不是正确标签，不在正确标签的支配队列中，但在正确标签的候选队列中
        # 更新支配队列
        if predict_label[i] != i and \
                real_predict_label not in d_label_queue and \
                real_predict_label in c_label_queue:
            dominant_queue[current_labels[i]] = enqueue(d_label_queue, real_predict_label)
