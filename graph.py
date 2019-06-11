# -*- coding:utf-8 -*-
"""
@project: untitled
@author: KunJ
@file: graph.py
@ide: Pycharm
@time: 2019-06-10 21:52:01
@month: Jun
"""

# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，
# 也就是说它在当前版本中不是语言标准，那么我们如果想要使用的话就要从__future__模块导入
from __future__ import print_function  # print()函数
import keras
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K

print("=================================定义基本的图卷积类===================================")
class GraphConvolution(Layer):
    """Keras自定义层要实现build方法、call方法和compute_output_shape(input_shape)方法(或者get_output_shape_for(input_shape)方法
    Basic graph convolution layer as in https://arxiv.org/abs/1609.02907
    """
    # 构造函数
    def __init__(self,
                 units,
                 support=1, # 支持多个图情况(即input = [feature][adj[i]]
                 activation=None, # 激活函数名称
                 use_bias=True, # 如果存在偏置
                 kernel_initializer='glorot_uniform',# eg. 'kernel_initializer'
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            # pop()函数用于删除列表中某元素，并返回该元素的值
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # 施加在权重上的正则项
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # 施加在偏置向量上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # 施加在输出上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # 对主权重矩阵进行约束
        self.kernel_constraint = constraints.get(kernel_constraint)
        # 对偏置向量进行约束
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.support = support

        assert support >= 1 # 图的个数大于1(即邻接矩阵个数大于1)


    def compute_output_shape(self, input_shapes):
        """
        计算输出的形状
        如果自定义层更改了输入张量的形状，则应该在这里定义形状变化的逻辑
        让Keras能够自动推断各层的形状
        :param input_shapes:
        :return:
        """
        # 特征矩阵形状
        features_shape = input_shapes[0]

        # 输出形状为(批大小, 输出维度)
        output_shape = (features_shape[0], self.units)

        return output_shape  # (batch_size, output_dim)


    def build(self, input_shapes):
        """
        定义层中的参数
        :param input_shapes: 输入张量的形状
        :return: 参数
        """
        # 特征矩阵形状
        features_shape = input_shapes[0]
        assert len(features_shape) == 2

        # 特征维度
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # 如果存在偏置
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # 必须设定self.bulit = True
        # 可以通过super(GraphConvolution, self).build()实现
        self.built = True

    def call(self, inputs, mask=None):
        """
        编写层的功能逻辑
        :param inputs: 输入张量
        :param mask:
        :return:
        """
        features = inputs[0]  # 特征
        A = inputs[1:]  # 对称归一化的邻接矩阵

        # 多个图的情况
        supports = list()
        for i in range(self.support):
            # A * X
            supports.append(K.dot(A[i], features))
        # 将多个图的结果按行拼接
        supports = K.concatenate(supports, axis=1)
        # A * X * W
        output = K.dot(supports, self.kernel)
        if self.bias:
            # A * X * W + b
            output += self.bias
        return self.activation(output)

    # 定义当前层的配置信息
    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,

                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),

                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),

                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
