# -*- coding: utf-8 -*-
"""
   File Name：     backbone
   Description : 骨干网络
   Author :       mick.yi
   date：          2018/12/21
"""

import keras
from keras import layers
from keras_applications.resnet50 import identity_block, conv_block


def resnet50(inputs):
    # Determine proper input shape
    bn_axis = 3

    # x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = layers.Conv2D(64, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      name='conv1')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 64], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 64], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 512], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 512], stage=5, block='c')

    # # 确定精调层
    # no_train_model = Model(inputs=img_input, outputs=x)
    # for l in no_train_model.layers:
    #     if isinstance(l, layers.BatchNormalization):
    #         l.trainable = True
    #     else:
    #         l.trainable = False

    # model = Model(input, x, name='resnet50')
    x = layers.GlobalAveragePooling2D()(x)
    # # 新增一个全连接层降维
    # x = layers.Dense(units=512)(x)
    return x


def cifar_base_model(inputs):
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    return x
