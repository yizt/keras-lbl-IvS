# -*- coding: utf-8 -*-
"""
   File Name：     pipeline
   Description :   处理流程
   Author :       mick.yi
   date：          2019/1/3
"""
import tensorflow as tf
import numpy as np
import h5py
import keras
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import sys
import os
import time
import argparse

if __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "keras-lbl-IvS"

from .config import current_config as config
from .utils import random_select, get_weights, update_weights, update_queue
from .faiss_utils import get_index, update_multi
from .reader import get_mslm_infos, load_img
from .layers import DenseWithDPSoftmaxLoss


def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    K.set_session(session)


def generator(images_info, label_id_dict, dominant_queue, num_class, batch_size):
    """
    训练样本生成器
    :param images_info: 图像的元数据信息
    :param label_id_dict: 类别和对应的图像id字典
    :param dominant_queue: 支配队列; 类别和对应的支配类别
    :param num_class: 类别数
    :param batch_size: batch_size
    :return:
    """
    while True:
        # 两级采样，首先采样batch_size/2个正类别;然后随机采样支配类别
        sample_labels = np.random.choice(num_class, batch_size // 2, replace=False)  # 无放回抽样

        selected_labels = set(sample_labels)  # 保存当前选中类别set
        selected_image_labels = []  # 保存当前step选中的图像和对应的类别标签
        # 首先选择正原型
        for label in sample_labels:
            selected_image_id = random_select(label_id_dict[label])
            selected_image_labels.append([selected_image_id, label])

        # 再选择相关的支配原型，直到mini-batch大小
        while len(selected_image_labels) < batch_size:
            # 随机采样当前正原型
            label = random_select(sample_labels)
            # 随机选择支配类别,不能是之前已经选择过的
            dq_label = random_select(dominant_queue[label])
            while dq_label in selected_labels:
                dq_label = random_select(dominant_queue[label])
            selected_labels.add(dq_label)
            # 选择支配类别的图像
            selected_image_id = random_select(label_id_dict[dq_label])
            selected_image_labels.append([selected_image_id, dq_label])
        # 当前选中标签
        selected_image_labels = np.asarray(selected_image_labels)  # 转为numpy数组
        current_selected_labels = selected_image_labels[:, 1]
        current_weights = get_weights(config.pw_h5_file[config.prototype_weights_dataset],
                                      current_selected_labels)

        # 加载图像
        images = [load_img(images_info[image_id]['img_path']) for image_id, label in selected_image_labels]
        images = np.asarray(images)
        # 返回当前mini-batch
        current_selected_labels = np.expand_dims(current_selected_labels, axis=1)
        # print("current_selected_labels.shape:{}".format(current_selected_labels.shape))
        # print("images.shape:{},type(images):{}".format(images.shape, type(images)))
        yield [images,
               current_weights,
               current_selected_labels], np.arange(batch_size)  # 标签类别永远是0~batch_size-1


def init_queue(index, weights_set, num_class, dq_num, cq_num):
    """
    初始化候选队列和支配队列
    :param index:
    :param weights_set: h5py dataset对象
    :param num_class:
    :param dq_num:
    :param cq_num:
    :return:
    """
    data, candidate_label_idx = index.search(weights_set[:num_class], cq_num)
    dominant_label_idx = candidate_label_idx[:, :dq_num]  # 候选队列包含支配队列

    # 转为字典类型
    dominant_queue = dict(enumerate(dominant_label_idx))
    candidate_queue = dict(enumerate(candidate_label_idx))

    return dominant_queue, candidate_queue


def init_prototype(images_info, label_id_dict, num_class):
    inputs = Input(batch_shape=(config.batch_size,) + config.input_shape)
    features = config.backbone(inputs)
    model = Model(inputs, features)
    model.load_weights(config.backbone_weights, by_name=True)
    # 原型权重一份放到hdf5，一份存放到faiss中(faiss中保留的不是精准的)
    f_label = h5py.File(config.prototype_weights_hdf5, 'w')
    label_feature = f_label.create_dataset(config.prototype_weights_dataset,
                                           shape=(num_class, 512), dtype='f')

    # 逐个类别处理
    for label in range(num_class):
        # 获取某个label的所有图像，并使用模型预测图像的特征，最后求均值作为label的原型权重
        image_ids = label_id_dict[label]  # 图像id
        images = [load_img(images_info[image_id]['img_path']) for image_id in image_ids]  # 图像数据
        features = model.predict(np.asarray(images))  # 输出特征
        features = keras.utils.np_utils.normalize(features)  # 归一化
        features = np.mean(features, axis=0)  # 求均值
        features = keras.utils.np_utils.normalize(features)  # 再次归一化; 是二维的

        # 赋值给hdf5
        label_feature[label] = features[0]  # (1,d) 转为 (d,)
        # 每1w次，刷写到磁盘
        if label % 500 == 0:
            f_label.flush()
            print("{} init_prototype 完成：{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                   label))
    # 最后关闭文件
    f_label.close()


def get_prototype(deal_batch=1000):
    f_label = h5py.File(config.prototype_weights_hdf5, 'r+')
    dset = f_label[config.prototype_weights_dataset]
    length = len(dset)
    index = get_index(512)
    # 逐个类别处理,更新faiss index
    for batch_no in range(length // deal_batch):
        start = batch_no * deal_batch
        end = (batch_no + 1) * deal_batch
        features = dset[start:end]
        update_multi(index, features, np.arange(start, end))
    # 处理不能整除的情况
    if not length % deal_batch == 0:
        start = length - length % deal_batch
        end = length
        features = dset[start:end]
        update_multi(index, features, np.arange(start, end))
    return f_label, index


def label_id_map(images_info, num_class):
    """
    将图像按照类别分组
    :param images_info: 图像字典{'img_path': 图像路径,'label': 类别,'img_id':图像id}
    :param num_class: 类别数
    :return:
    """
    # 初始化
    label_id_dict = dict()
    for i in range(num_class):
        label_id_dict[i] = []

    # 逐个图像归类
    for i in range(len(images_info)):
        label = int(images_info[i]['label'])
        img_id = images_info[i]['img_id']
        label_id_dict[label].append(img_id)

    return label_id_dict


class ExportWeights(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        layer = self.model.layers[-1]

        trained_weights, current_trained_labels, y_pred = layer.get_weights()[:3]
        trained_weights = np.transpose(trained_weights)
        current_trained_labels = np.asarray(current_trained_labels[:, 0], dtype=np.int)
        # print("\n input_weights:{}".format(config.current_input[1][0][:10]))
        # print("\n trained_weights:{}".format(trained_weights[0][:10]))
        # current_selected_labels = config.current_selected_labels
        # print("current_selected_labels:{}".format(current_selected_labels))
        # print("current_trained_labels:{}".format(current_trained_labels))
        update_multi(config.index, trained_weights, current_trained_labels)  # 更新faiss index
        update_weights(config.pw_h5_file,
                       config.pw_h5_file[config.prototype_weights_dataset],
                       trained_weights,
                       current_trained_labels)
        # 以下更新支配队列,根据预测结果更新支配队列;
        # todo : 需要使用更加高效的方式来获取输出结果，使用predict相当于又做了一次前向传播(已解决，使用权重保存)
        # y_pred = self.model.predict(config.current_input)
        # print("y_pred:{}".format(y_pred))
        update_queue(config.dominant_queue,
                     config.candidate_queue,
                     y_pred,
                     current_trained_labels)


def get_call_back():
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/lbl-IvS.{epoch:03d}.h5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=False)

    # 验证误差没有提升
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=np.sqrt(0.1),
                                   cooldown=1,
                                   patience=1,
                                   min_lr=0)

    log = TensorBoard(log_dir='log')

    export_weights = ExportWeights()

    return [checkpoint, lr_reducer, export_weights, log]


def main(args):
    K.clear_session()
    set_gpu_growth()
    # 获取图像元数据信息
    images_info, label_set = get_mslm_infos(config.ms1m_annotation_file, config.ms1m_img_dir)
    # 过滤小于num_classes的类别;测试用
    images_info = filter(lambda x: int(x['label']) < config.num_classes, images_info)
    images_info = dict(enumerate(images_info))  # 转为字典

    label_id_dict = label_id_map(images_info, config.num_classes)  # 类别对应的

    # 初始化原型权重
    if 'init' in args.stages:
        init_prototype(images_info,
                       label_id_dict,
                       config.num_classes)
        print("初始化原型权重完成... ...")

    # 训练阶段
    if 'train' in args.stages:
        config.pw_h5_file, config.index = get_prototype(1000)
        print("获取原型权重完成... ...")
        # current_weights = get_weights(config.pw_h5_file[config.prototype_weights_dataset],
        #                               np.arange(config.batch_size))
        # print(current_weights.shape)
        # print(type(current_weights))
        # print(current_weights[0])

        # 初始化队列
        config.dominant_queue, config.candidate_queue = init_queue(config.index,
                                                                   config.pw_h5_file[config.prototype_weights_dataset],
                                                                   config.num_classes,
                                                                   config.dominant_queue_num,
                                                                   config.candidate_queue_num)
        print("初始化队列完成... ...")
        # 构建模型
        inputs = Input(batch_shape=(config.batch_size,) + config.input_shape)
        weights_inputs = Input(batch_shape=(config.batch_size, 512))
        label_inputs = Input(batch_shape=(config.batch_size, 1))  # 至少是二维的,
        features = config.backbone(inputs)
        dense = DenseWithDPSoftmaxLoss(config.batch_size)  # batch-size当做类别数
        outputs = dense([features, weights_inputs, label_inputs])
        m = Model([inputs, weights_inputs, label_inputs], outputs)
        m.summary()
        m.load_weights(config.backbone_weights, by_name=True)
        m.compile(loss=dense.loss, optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0005), metrics=['accuracy'])

        # 训练模型
        print("开始训练模型... ...")
        gen = generator(images_info,
                        label_id_dict,
                        config.dominant_queue,
                        config.num_classes,
                        config.batch_size)

        m.fit_generator(gen,
                        callbacks=get_call_back(),
                        steps_per_epoch=len(images_info) / 4 // config.batch_size,
                        epochs=10,
                        use_multiprocessing=False,
                        verbose=1,
                        validation_data=next(gen)
                        )

        # 最后关闭hdf5
        config.pw_h5_file.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--stages", type=str, nargs='+', default=['train'], help="stage: init、train")
    argments = parse.parse_args(sys.argv[1:])
    # print(argments)
    # useage: python pipeline.py --stages init train | python pipeline.py --stages train
    main(argments)
