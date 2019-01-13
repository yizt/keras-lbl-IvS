# -*- coding: utf-8 -*-
"""
   File Name：     reader
   Description :  读取图像信息
   Author :       mick.yi
   date：          2018/12/26
"""
import os
import codecs
import matplotlib.pyplot as plt


def get_mslm_infos(annotation_file, img_dir):
    """
    读取mslm数据集信息
    :param annotation_file: 标注文件路径
    :param img_dir: 图像存放路径
    :return:
    """
    with codecs.open(annotation_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    img_infos = []

    label_set = set()
    for id, line in enumerate(lines):
        img_name, label = line.split('\t')
        img_info = dict()
        img_info['img_path'] = os.path.join(img_dir, img_name)
        img_info['label'] = label
        img_info['img_id'] = id   # 增加图像id编号
        img_infos.append(img_info)
        label_set.add(label)

    return img_infos, label_set


def load_img(img_path):
    img = plt.imread(img_path)
    return img[:, :, :3]
