import csv
import sys
import numpy as np
import math
from numpy import linalg as LA
import random
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def load_csv(input_file):
    file = open(input_file, 'rU')
    data = list(csv.reader(file, delimiter=','))
    data = np.asarray(data)
    return data



def HC(data, method):
    #data = sub_data(data)
    xy = np.zeros((len(data), 2))
    xy[:, 0] = data[:, 2]
    xy[:, 1] = data[:, 3]
    Z = linkage(xy, method)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    dendrogram(Z)
    plt.show()
    return Z


def HC_cluster(data, method, k):
    xy = np.zeros((len(data), 2))
    xy[:, 0] = data[:, 2]
    xy[:, 1] = data[:, 3]
    Z = linkage(xy, method)
    cluster = fcluster(Z, k, criterion='maxclust')
    return cluster


def trans_HC(data, cluster, k):
    cluster = np.asarray(cluster)
    tran_cluster = np.zeros((len(data), k))
    for i in range(k):
        tran_cluster[cluster == (i+1), i] = 1
    return tran_cluster


def HC_wcssd(data, cluster, k):
    k_means = np.zeros((k, 2))
    xy = np.zeros((len(data), 2))
    xy[:, 0] = data[:, 2]
    xy[:, 1] = data[:, 3]
    distance = 0
    for j in range(k):
        chose_xy = xy[cluster[:, j] == 1]
        k_means[j] = np.mean(chose_xy, axis=0)
        distance += sum((LA.norm(chose_xy - k_means[j], axis=1)) ** 2)
    return distance



def C_3(data, k):
    data = sub_data(data)
    method = ['single', 'complete', 'average']
    wcssd = np.zeros((len(k), 1))
    sc = np.zeros((len(k), 1))
    for i in range(len(method)):
        Z = HC(data, method[i])
        for j in range(len(k)):
            cluster = fcluster(Z, k[j], criterion='maxclust')
            cluster = trans_HC(data, cluster, k[j])
            wcssd[j] = HC_wcssd(data, cluster, k[j])
            sc[j] = silh_coeff(data, cluster)
        print method[i]
        print 'wcssd', wcssd
        print 'sc', sc
    return


def main():
	data = load_csv(data_set)
	k = 32
	c = HC_cluster(data, 'single', k)
    m = trans_HC(data, c, 32)
    print get_NMI(data, m)
    c = HC_cluster(data, 'complete', k)
    m = trans_HC(data, c, 32)
    print get_NMI(data, m)
    c = HC_cluster(data, 'average', k)
    m = trans_HC(data, c, 32)
    print get_NMI(data, m)

 main()


