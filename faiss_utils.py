# -*- coding: utf-8 -*-
"""
   File Name：     faiss_utils
   Description :   faiss工具类
   Author :       mick.yi
   date：          2019/1/4
"""
import faiss
import numpy as np


def get_index(dimension):
    sub_index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(sub_index)
    return index


def update_multi(index, vectors, ids):
    """

    :param index:
    :param vectors:
    :param ids:
    :return:
    备注：ValueError: array is not C-contiguous
    """
    idx = np.argsort(ids)
    # 先删除再添加
    index.remove_ids(ids[idx])
    index.add_with_ids(vectors[idx], ids[idx])


def update_one(index, vector, label_id):
    vectors = np.expand_dims(vector, axis=0)
    ids = np.array([label_id])
    update_multi(index, vectors, ids)
