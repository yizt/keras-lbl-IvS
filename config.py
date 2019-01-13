# -*- coding: utf-8 -*-
"""
   File Name：     config
   Description :  配置文件
   Author :       mick.yi
   date：          2019/1/3
"""
from .backbone import resnet50


class Config(object):
    #################
    # basenet 阶段
    #################
    # ms1m人脸数据标注和图像图论
    ms1m_annotation_file = '/home/dataset/face_recognize/ms1m_112_112.label'
    ms1m_img_dir = '/home/dataset/face_recognize/ms1m_112_112'
    # 输入信息
    input_shape = (112, 112, 3)
    num_classes = 2000  # 测试时可以调小点 85164

    # 网络结构信息
    def backbone(self, inputs):
        return resnet50(inputs)

    # 训练参数
    batch_size = 48

    lr = 0.1
    learning_rate_schedule = {
        0: 1 * lr,
        160000: 0.1 * lr,
        240000: 0.01 * lr,
        280000: 0.001 * lr
    }

    backbone_weights = '/tmp/docface.basenet.002.h5'

    # 其它参数信息
    prototype_weights_hdf5 = '/tmp/prototype_weights.hdf5'
    prototype_weights_dataset = 'prototype_weights_set'
    pw_h5_file = None  # hdf5 File文件

    index = None  # 保存原型类别的faiss索引

    dominant_queue = None  # 支配队列
    candidate_queue = None  # 候选队列

    dominant_queue_num = 10  # 支配队列大小
    candidate_queue_num = 30  # 候选队列大小

    # 评估阶段参数
    lbl_weights = '/tmp/lbl-IvS.010.h5'


# 当前配置
current_config = Config()
