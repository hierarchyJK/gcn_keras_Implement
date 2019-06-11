# -*- coding:utf-8 -*-
"""
@project: untitled
@author: KunJ
@file: utils.py
@ide: Pycharm
@time: 2019-06-10 14:29:44
@month: Jun
"""

# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，

# 也就是说它在当前版本中不是语言标准，那么我们如果想要使用的话就要从__future__模块导入

from __future__ import print_function  # print()函数
import scipy.sparse as sp  # python中稀疏矩阵相关库
import numpy as np  # python中操作数组的函数
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence# 稀疏矩阵中查找特征值/特征向量的函数(由于这adj是一个对称矩阵，我们用eigsh)
from scipy.sparse.linalg import eigsh


def encode_onehot(labels):
    """
    将标签转换为one-hot编码形式
    :param labels: 标签--list
    :return: 标签的one-hot形式
    """
    # set()函数创建一个不重复元素集合, set()作用于list后，结果会自动排序
    classes = set(labels)

    # np.identity()函数创建方阵，返回主对角线元素为1，其余元素为0的数组
    # enumerate()函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列,同时列出数据下标和数据，一般用在for循环中
    classes_dict = {c: np.identity(len(classes))[i,:] for i,c in enumerate(classes)}

    # map()函数根据提供的函数对指定序列做映射
    # map(function, iterable)
    # 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

    return labels_onehot


#

def load_data(path="/home/jiangkun/gcn-master/data/cora/", dataset="cora"):
    """加载数据,只加载cora数据,Load citation network dataset (cora only for now)"""

    # str.format()函数用于格式化字符串
    print('Loading {} dataset...'.format(dataset))

    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # fname：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    print(len(idx_features_labels[:,-1]))
    print('标签:',set(idx_features_labels[:,-1]))

    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用(行索引、列索引)和值表示矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    print('\n样本的特征:\n',features)
    print('feature.shape:', features.shape)

    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph

    # 样本的id数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    print('\n节点数:',len(idx))

    # 由样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx)}
    print("\n样本id:样本索引\n", idx_map)

    # 样本id之间的引用关系数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    print('\n样本id之间的引用关系:\n', edges_unordered)

    # 将样本id之间的引用关系用样本索引之间的关系表示
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    print('\n样本索引之间的关系:\n', edges)
    print('\n样本标签的one-hot形式:\n',labels)
    print('edges.shape:', edges.shape)

    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，注意: 这里根据数据内容构建的是非对称邻接矩阵
    # coo_matrix((data, (row, col)), shape, dtype)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    print("\n图的邻接矩阵，用坐标形式的稀疏矩阵表示\n", adj)

    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print('\n对称邻接矩阵:\n',adj)

    # 打印消息：数据集有多少个节点、多少条边、每个样本有多少维特征
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    print('\nfeatures.todense():\n',features.todense()) # 将稀疏矩阵还原成密集矩阵用todense()方法
    print(features.todense().shape)
    print('\nadj.todense():\n',adj.todense()) # 将稀疏矩阵还原成密集矩阵用todense()方法
    print(adj.todense().shape)
    print('label.shape', labels.shape)
    # 返回特征的密集矩阵表示、邻接矩阵和标签的one-hot编码
    return features.todense(), adj, labels
feature, adj, labels = load_data()

print("------------------------------------------------------------------------------------------")

def normalize_adj(adj, symmetric=True):
    """
    对邻接矩阵进行归一化处理
    :param adj: 邻接矩阵(密集矩阵)
    :param symmetric: 是否对称
    :return: 归一化后的邻接矩阵
    """
    # 如果邻接矩阵为对称矩阵，得到对称归一化邻接矩阵
    # D^(-1/2) * A * D^(-1/2)
    if symmetric:
        # A.sum(axis=1)：计算矩阵的每一行元素之和，得到节点的度矩阵D
        # np.power(x, n)：数组元素求n次方，得到D^(-1/2)
        # sp.diags()函数根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素\
        d = sp.diags(np.power(np.array(adj.sum(axis=1)), -1/2).flatten(), 0)
        print("对角矩阵:\n", d.todense())

        # tocsr()函数将矩阵转化为压缩稀疏行矩阵  D^(-1/2) * A * D^(-1/2)
        a_norm = d.dot(adj).dot(d).tocsr()

    # 如果邻接矩阵不是对称矩阵，得到随机游走正则化拉普拉斯算子
    # D^(-1) * A
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        # D^(-1) * A
        a_norm = d.dot(adj).tocsr()
    return a_norm

print("------------------------------------------------------------------------------------------")

#
def preprocess_adj(adj, symmetric=True):
    """
    在邻接矩阵中加入自连接(因为自身信息很重要)
    :param adj: 原始的邻接矩阵
    :param symmetric: 是否对称
    :return: 加入自连接后的邻接矩阵
    """
    adj = adj + sp.eye(adj.shape[0])
    print(adj)
    # 对加入自连接的邻接矩阵进行对称归一化处理
    adj = normalize_adj(adj, symmetric)

    return adj
adj = preprocess_adj(adj)

print("----------------------------------------构造样本掩码--------------------------------------------------")

def sample_mask(idx, nums_sample):
    """
    构造样本掩码
    :param idx: 有标签样本的索引列表
    :param nums_sample: 所有样本数量
    :return: 布尔类型数组，其中有标签样本所对应的位置为True，无标签样本所对应的位置为False
    """
    # np.zeros()函数创建一个给定形状和类型的用0填充的数组
    mask = np.zeros(nums_sample)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

print("---------------------------------------数据集划分---------------------------------------------------")

def get_splits(y):
    """
    数据集划分
    :param y: labels
    :return:训练集样本标签、验证集样本标签、测试集样本标签、训练集索引列表、验证集索引列表、测试集索引列表、训练数据的样本掩码
    """
    # 训练集索引列表
    idx_train = range(140)
    # 验证集索引列表
    idx_val = range(200, 500)
    # 测试集索引列表
    idx_test = range(500, 1500)

    # 训练集样本标签
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    # 验证集样本标签
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_val[idx_val] = y[idx_val]
    # 测试集样本标签
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_test[idx_test] = y[idx_test]

    # 训练数据的样本掩码
    train_mask = sample_mask(idx_train, y.shape[0])

    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(labels)
print('y_train =\n', y_train, y_train.shape)
print('y_val =\n', y_val, y_val.shape)
print('y_test =\n', y_test, y_test.shape)
print('idx_train =\n',idx_train)
print('idx_val =\n', idx_val)
print('idx_test=\n', idx_test)
print('train_mask =\n', train_mask, 'len(train_mask)=',len(train_mask))
count=0
countall = 0
for i in y_test:
    countall +=1
    if 1 in i:
       count +=1
print('count =',count)
print('countall = ',countall)

print("*************************************定义分类交叉熵***************************************")
def categorical_crossentropy(preds, labels):
    """
    定义分类交叉熵
    :param preds:模型对样本的输出数组
    :param labels:样本的one-hot标签数组
    :return:样本的平均交叉熵损失
    """
    # np.extract(condition, x)函数，根据某个条件从数组中抽取元素
    # np.mean()函数默认求数组中所有元素均值
    return np.mean(-np.log(np.extract(labels, preds)))


print('*************************************定义准确率函数*****************************************')
def accuracy(preds, labels):
    """
    定义准确率函数
    :param preds: 模型对样本的输出数组
    :param labels: 样本的one-hot标签数组
    :return: 样本的平均准确率
    """
    # np.argmax(x, axis)函数取出x中元素最大值所对应的索引(axis=0:column, axis=1:row)
    # np.equal(x1, x2)函数用于在元素级比较两个数组是否相等
    return np.mean(np.equal(np.argmax(labels,axis=1), np.argmax(preds,axis=1)))

print('***************************************评估样本划分的损失函数和准确率***************************')


def evaluate_preds(preds, labels, indices):
    """
    评估样本划分的损失函数和准确率
    :param preds:对于样本的预测值
    :param labels:样本的标签one-hot向量
    :param indices:样本的索引集合
    :return:交叉熵损失函数列表、准确率列表
    """

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        # 计算每一个样本划分的交叉熵损失函数
        split_loss.append(categorical_crossentropy(preds[idx_split],y_split[idx_split]))

        # 计算每一个样本划分的准确率
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

print('*************************************对图拉普拉斯矩阵进行归一化处理*********************************')
def normalized_laplacian(adj, symmetric=True):
    """
    对图拉普拉斯矩阵进行归一化处理
    :param adj: 原始的邻接矩阵
    :param symmetric: 是否对称
    :return:对称规范化的图拉普拉斯矩阵
    """
    # 对称归一化的邻接矩阵，D ^ (-1/2) * A * D ^ (-1/2)
    adj_normalized = normalize_adj(adj, symmetric)

    # 得到对称规范化的图拉普拉斯矩阵，L = I - D ^ (-1/2) * A * D ^ (-1/2)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized

    return laplacian


def rescale_laplacian(laplacian):
    """
    重新调整对称归一化的图拉普拉斯矩阵，得到其简化版本
    :param laplacian: 未调整的图拉普拉斯矩阵
    :return: 重新调整后的图拉普拉斯矩阵
    """
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        # 计算对称归一化图拉普拉斯矩阵的最大特征值
        largest_eigval = eigsh(laplacian, k=1, which='LM', return_eigenvectors=False)[0] # 返回特征值和特征向量

    # 如果计算过程不收敛
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    # 调整后的对称归一化图拉普拉斯矩阵，L~ = 2 / Lambda * L - I

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])

    return scaled_laplacian

print('****************************************计算直到k阶的切比雪夫多项式**********************************')
def chebyshev_polynomial(X, k):
    """
    计算直到k阶的切比雪夫多项式
    :param X: L~ --> scaled_laplacian
    :param k: order(degree)
    :return:
    """
    # 返回一个稀疏矩阵列表
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())  # T0(X) = I
    T_k.append(X)  # T1(X) = L~

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        """
        定义切比雪夫递归公式
        :param T_k_minus_one: T(k-1)(L~)
        :param T_k_minus_two: T(k-2)(L~)
        :param X: L~
        :return: Tk(L~)
        """
        # 将输入转化为csr矩阵（压缩稀疏行矩阵）
        X_ = sp.csr_matrix(X, copy=True)
        # 递归公式：Tk(L~) = 2L~ * T(k-1)(L~) - T(k-2)(L~)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    # 返回切比雪夫多项式列表
    return T_k



def sparse_to_tuple(sparse_mx):
    """
    将稀疏矩阵转化为元组表示
    :param sparse_mx: 稀疏矩阵
    :return:
    """
    if not sp.isspmatrix_coo(sparse_mx):
        # 将稀疏矩阵转化为coo矩阵形式
        # coo矩阵采用三个数组分别存储行、列和非零元素值的信息
        sparse_mx = sparse_mx.tocoo()
    # np.vstack()函数沿着数组的某条轴堆叠数组
    # 获取非零元素的位置索引
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # 获取矩阵的非零元素
    values = sparse_mx.data
    # 获取矩阵的形状
    shape = sparse_mx.shape
    return coords, values, shape
