# -*- coding: utf-8 -*-
"""
   File Name：     evaluate
   Description :   lbl-IvS评估
   Author :       mick.yi
   date：          2019/1/10
"""
import keras
from keras import Input
from keras.layers import Dense
from keras.models import Model
import numpy as np
import sys
import os
import h5py

if __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "keras-lbl-IvS"

from .config import current_config as config
from .reader import get_mslm_infos, load_img


def gen_data(images_info, batch_size):
    all_size = len(images_info)
    while True:
        batch_idx = np.random.choice(all_size, batch_size, replace=False)
        images = []
        labels = []
        for i in batch_idx:
            images.append(load_img(images_info[i]['img_path']))
            labels.append(int(images_info[i]['label']))
        yield np.asarray(images), np.asarray(labels)


def main():
    # 构建模型
    inputs = Input(batch_shape=(config.batch_size,) + config.input_shape)
    features = config.backbone(inputs)
    dense = Dense(config.num_classes,
                  use_bias=False,
                  activation='softmax')  # 增加最后一层分类层;这里需要使用softmax激活，默认不是fromlogits
    outputs = dense(features)
    m = Model(inputs, outputs)
    m.compile(optimizer='SGD',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
    m.load_weights(config.lbl_weights, by_name=True)

    # 设置分类层的权重
    f_label = h5py.File(config.prototype_weights_hdf5, 'r')
    weights = f_label[config.prototype_weights_dataset]  # HDF5 dataset object
    m.layers[-1].set_weights([np.transpose(weights[:])])  # 使用weights[:]转为numpy
    m.summary()

    # 加载数据
    images_info, label_set = get_mslm_infos(config.ms1m_annotation_file, config.ms1m_img_dir)
    # 过滤小于num_classes的类别;测试用
    images_info = filter(lambda x: int(x['label']) < config.num_classes, images_info)

    # 评估预测
    scores = m.evaluate_generator(gen_data(list(images_info), 200),
                                  steps=10,
                                  verbose=1)
    m.predict_generator
    print(scores)


if __name__ == '__main__':
    main()
