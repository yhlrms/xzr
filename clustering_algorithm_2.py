#修改
from time import time
import numpy as np, matplotlib.pyplot as mp

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler  # 数据标准化
from itertools import cycle, islice

"""生成随机样本集"""
np.random.seed(0)  # 设定相同的随机环境，使每次生成的随机数相同

n_samples = 1500

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# 非均质分散的数据
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# 方差各异的团
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

"""设置聚类和绘图参数"""
mp.figure(figsize=(9 * 2 + 3, 12.5))
mp.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
plot_num = 1

default_base = {'quantile': .3,  # 分位数
                'eps': .3,  # DBSCAN同类样本间最大距离
                'damping': .9,  # 近邻传播的阻尼因数
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # 更新样本集特征对应的参数
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # 数据标准化
    X = StandardScaler().fit_transform(X)

    # 估计均值漂移的带宽
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # 层次聚类参数
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)  # 连接矩阵
    connectivity = 0.5 * (connectivity + connectivity.T)  # 使其对称化

    """创建各个聚类对象"""

    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('GaussianMixture', gmm),
    )

    """绘图"""
    for name, algorithm in clustering_algorithms:
        t0 = time()
        algorithm.fit(X)
        t1 = time()

        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        mp.subplot(2, 3, plot_num)
        if i_dataset == 1:  # 第0行打印标题
            mp.title(name, size=20)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        colors = np.append(colors, ["#000000"])  # 离群点（若有的话）设为黑色
        mp.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        mp.xlim(-2.5, 2.5)
        mp.ylim(-2.5, 2.5)
        mp.xticks(())
        mp.yticks(())
        mp.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                transform=mp.gca().transAxes, size=14, horizontalalignment='right')
        plot_num += 1
mp.show()
