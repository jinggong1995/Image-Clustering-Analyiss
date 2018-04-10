import csv
import sys
import numpy as np
import math
from numpy import linalg as LA
import random
from matplotlib import pyplot as plt
import matplotlib.cm as cm



def get_PCA(data, p):
    D = np.zeros((len(data), len(data[0, 2:])))
    X = np.zeros((len(data), len(data[0, 2:])))
    D[:, 0:] = data[:, 2:]
    for i in range(len(D[0])):
        X[:, i] = D[:, i] - np.mean(D[:, i])
    cov = np.cov(X, rowvar=False)
    value, vector = LA.eig(cov)
    index = np.argsort(-value)
    value = value[index]
    vector = vector[:, index]
    vector = vector[:, :p]
    vector = np.real(vector)
    new_vector = np.dot(X, vector)
    return vector, new_vector



def PCA_kmeans_cluster(data, p, k, index):
    k_means = data[index]
    distance = np.zeros((len(data), k))
    cluster = np.zeros((len(data), k))
    count = 0
    while (count < 50):
        cluster = np.zeros((len(data), k))
        for j in range(k):
            distance[:, j] = LA.norm(data - k_means[j], axis=1)
        min_distance = np.amin(distance, axis=1)
        for j in range(k):
            cluster[distance[:, j] <= min_distance, j] = 1
            k_means[j, :] = sum(data[distance[:, j] == min_distance]) / float(sum(cluster[:, j]))
        count += 1
    wcssd = sum(min_distance ** 2)
    return cluster, wcssd




def silh_PCA(data, cluster):
    S = np.zeros((len(data), 1))
    B = np.zeros((len(cluster[0]) - 1, 1))
    for i in range(len(data)):
        distance = LA.norm(data - data[i], axis=1)
        c_a = cluster[:, cluster[i] == 1]
        c_b = cluster[:, cluster[i] != 1]
        A = np.mean(distance[c_a[:, 0] == 1])
        for j in range(len(c_b[0])):
            B[j] = np.mean(distance[c_b[:, j] == 1])
        S[i] = (min(B) - A) / max(A, min(B))
    ss = np.mean(S)
    return ss




def main():
	data = load_csv('digits-raw.csv')
    p = 10
    k = [2, 4, 8, 16, 32]
    vector, new_vector = get_PCA(data, p)
    index = random.sample(list(range(len(data))), 2)
    vector, new_vector = get_PCA(data, p)
    cluster, wcssd = PCA_kmeans_cluster(new_vector, p, 2, index)
    print get_NMI(data, cluster)
    print silh_PCA(data, cluster)


main()


