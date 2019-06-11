# -*- coding:utf-8 -*-
"""
@project: untitled
@author: KunJ
@file: train.py
@ide: Pycharm
@time: 2019-06-11 11:02:54
@month: Jun
"""

from __future__ import print_function
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from graph import GraphConvolution
from utils import *
import time
import keras
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 超参数
# Define parameters
DATASET = 'cora'
# 过滤器
FILTER = 'localpool'  # 'chebyshev'两种滤波器
# 最大多项式的度
MAX_DEGREE = 2  # maximum polynomial degree
# 是否对称正则化
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
# 迭代次数
NB_EPOCH = 20000
# 提前停止参数
PATIENCE = 10  # early stopping patience

# 加载数据
# Get data
X, A, y = load_data(dataset=DATASET)  # 特征(2708, 1433)、邻接矩阵(2708, 2708)、标签(2708, 7)

# 训练集样本标签、验证集样本标签、测试集样本标签、训练集索引列表、验证集索引列表、测试集索引列表、训练数据的样本掩码
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

# 对特征进行归一化处理
# Normalize X
X /= X.sum(1).reshape(-1, 1)

# 当过滤器为局部池化过滤器时
if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    # 加入自连接的邻接矩阵
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1

    # 特征矩阵和邻接矩阵
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

# 当过滤器为切比雪夫多项式时
elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')

    # 对拉普拉斯矩阵进行归一化处理，得到对称规范化的拉普拉斯矩阵
    L = normalized_laplacian(A, SYM_NORM)

    # 重新调整对称归一化的图拉普拉斯矩阵，得到其简化版本
    L_scaled = rescale_laplacian(L)

    # 计算直到MAX_DEGREE阶的切比雪夫多项式
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)

    support = MAX_DEGREE + 1

    # 特征矩阵、MAX_DEGREE阶的切比雪夫多项式列表
    graph = [X] + T_k  # 列表相加
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
else:
    raise Exception('Invalid filter type.')

# shape为形状元组，不包括batch_size
# 例如shape=(32, )表示预期的输入将是一批32维的向量
X_in = Input(shape=(X.shape[1],))
# 定义模型架构
# 注意：我们将图卷积网络的参数作为张量列表传递
# 更优雅的做法需要重写Layer基类
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + G)
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H] + G)

# 编译模型
# Compile model
model = Model(inputs=[X_in] + G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01),metrics=['acc'])
model.summary()
model.fit(graph, y_train,sample_weight=train_mask,batch_size=A.shape[0],epochs=350,shuffle=False,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1e-2, patience=10,verbose=1)])
# 训练过程中的辅助变量
# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999
# # 训练模型
# # Fit
# for epoch in range(1, NB_EPOCH + 1):
#     # 统计系统时钟的时间戳
#     # Log wall-clock time
#     t = time.time()
#
#     # 每一次迭代过程
#     # Single training iteration (we mask nodes without labels for loss calculation)
#     model.fit(graph, y_train, sample_weight=train_mask, # 向sample_weight参数传递train_mask用于样本掩码
#               batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
#
#     # 预测模型在整个数据集上的输出
#     # Predict on full dataset
#     preds = model.predict(graph, batch_size=A.shape[0])
#
#     # 模型在验证集上的损失和准确率
#     # Train / validation scores
#     train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])
#
#     print("Epoch: {:04d}".format(epoch),
#           "train_loss= {:.4f}".format(train_val_loss[0]),  # 在训练集上的损失
#           "train_acc= {:.4f}".format(train_val_acc[0]),  # 在训练集上的准确率
#           "val_loss= {:.4f}".format(train_val_loss[1]),  # 在验证集上的损失
#           "val_acc= {:.4f}".format(train_val_acc[1]),  # 在验证集上的准确率
#           "time= {:.4f}".format(time.time() - t))  # 本次迭代的运行时间
#
#     # 提取停止
#     # Early stopping
#     if train_val_loss[1] < best_val_loss:
#         best_val_loss = train_val_loss[1]
#         wait = 0
#     else:
#         # 当模型在测试集上的损失连续10次迭代没有优化时，则提取停止
#         if wait >= PATIENCE:
#             print('Epoch {}: early stopping'.format(epoch))
#             break
#         wait += 1
# 模型在测试集上的损失和准确率
# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))