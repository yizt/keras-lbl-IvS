# -*- coding: utf-8 -*-
"""
   File Name：     layers
   Description :   keras 层
   Author :       mick.yi
   date：          2019/1/2
"""
from keras.layers import Layer
from keras import backend as K


class DenseWithDPSoftmaxLoss(Layer):
    def __init__(self, num_class, m=0.35, scale=30, **kwargs):
        self.output_dim = num_class
        self.margin = m
        self.scale = scale
        # self.current_selected_labels = K.ones(shape=(num_class, 1))
        super(DenseWithDPSoftmaxLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),  # (嵌入维度,num_class)
                                      dtype=K.floatx(),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.current_selected_labels = self.add_weight(name='labels',
                                                       shape=(self.output_dim, 1),
                                                       initializer='glorot_normal',
                                                       trainable=False)
        self.y_pred = self.add_weight(name='pred',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=False)

    def call(self, inputs, **kwargs):
        # 将当前step类别的权重，赋值给tensor
        weights_assign_op = K.tf.assign(self.kernel,
                                        K.transpose(inputs[1]),
                                        name='assign_weights')
        label_assign_op = K.tf.assign(self.current_selected_labels, inputs[2], name='assign_labels')
        with K.tf.control_dependencies([weights_assign_op, label_assign_op]):
            self.x_norm = K.l2_normalize(inputs[0], axis=1)
            self.kernel_norm = K.l2_normalize(self.kernel, axis=0)
            self.logit = K.dot(self.x_norm, self.kernel_norm)
        return self.logit

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],
                self.output_dim)
        # return [(input_shape[0][0], self.output_dim),
        #         (self.output_dim, input_shape[0][1])]

    def loss(self, y_true, y_pred):
        # 首先将预测值保持到权重中
        pred_assign_op = K.tf.assign(self.y_pred,
                                     y_pred,
                                     name='assign_pred')
        with K.tf.control_dependencies([pred_assign_op]):
            y_true = y_true[:, 0]  # 非常重要，默认是二维的
            y_true_mask = K.one_hot(K.tf.cast(y_true, dtype='int32'), self.output_dim)
            cosine_m = y_pred - y_true_mask * self.margin  # cosine-m
            losses = K.sparse_categorical_crossentropy(target=y_true,
                                                       output=cosine_m * self.scale,
                                                       from_logits=True)

        return losses
