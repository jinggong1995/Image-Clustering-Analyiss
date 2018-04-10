import csv
import sys
import numpy as np
import math
from numpy import linalg as LA
import random
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def load_csv(input_file):
    file = open(input_file, 'rU')
    data = list(csv.reader(file, delimiter=','))
    data = np.asarray(data)
    return data



def k_mean_cluster(data, index, max_iter):
    xy = np.zeros((len(data), 2))
    xy[:, 0] = data[:, 2]
    xy[:, 1] = data[:, 3]
    k = len(index)
    k_means = xy[index]
    distance = np.zeros((len(data), k))
    cluster = np.zeros((len(data), k))
    count = 0
    while(count < max_iter):
        cluster = np.zeros((len(data), k))
        for j in range(k):
            distance[:, j] = LA.norm(xy - k_means[j], axis=1)
        min_distance = np.amin(distance, axis=1)
        for j in range(k):
            cluster[distance[:, j] <= min_distance, j] = 1
            k_means[j, :] = sum(xy[distance[:, j] == min_distance]) / float(sum(cluster[:, j]))
        count += 1
    wcssd = sum(min_distance**2)
    return cluster, wcssd



def silh_coeff(data, cluster):
    xy = np.zeros((len(data), 2))
    xy[:, 0] = data[:, 2]
    xy[:, 1] = data[:, 3]
    S = np.zeros((len(data), 1))
    B = np.zeros((len(cluster[0]) - 1, 1))
    for i in range(len(data)):
        distance = LA.norm(xy - xy[i], axis=1)
        c_a = cluster[:, cluster[i] == 1]
        c_b = cluster[:, cluster[i] != 1]
        A = np.mean(distance[c_a[:, 0] == 1])
        for j in range(len(c_b[0])):
            B[j] = np.mean(distance[c_b[:, j] == 1])
        S[i] = (min(B) - A) / max(A, min(B))
    ss = np.mean(S)
    return ss



def get_NMI(data, cluster):
    sum_cluster = np.sum(cluster, axis=0)
    N = len(data)
    H_cluster = - sum(sum_cluster/float(N) * np.log(sum_cluster/float(N)))
    sum_class = np.zeros((10, 1))
    for i in range(10):
        sum_class[i, 0] = sum(data[:, 1] == str(i))
    cal_class = sum_class
    cal_class[cal_class[:, 0] == 0, 0] = 1
    H_class = - sum(sum_class / float(N) * np.log(cal_class / float(N)))
    info = 0
    for i in range(len(cluster[0])):
        w = cluster[:, i]
        for j in range(10):
            w_c = sum(w[data[:, 1] == str(j)] == 1)
            if w_c != 0:
                info += w_c/float(N) * np.log(N * w_c / float(sum(w) * sum_class[j]))
    NMI = info / (H_cluster + H_class)
    return NMI



def main():


    if len(sys.argv) == 3:
        data_set = sys.argv[1]
        data = load_csv(data_set)
        k = int(sys.argv[2])
        index = random.sample(list(range(len(data))), k)
        cluster, wcssd = k_mean_cluster(data, index, 50)
        print 'WC-SSD', str(wcssd)
        ss = silh_coeff(data, cluster)
        print 'SC', str(ss)
        NMI = get_NMI(data, cluster)
        NMI = NMI[0]
        print 'NMI', str(NMI)

main()


