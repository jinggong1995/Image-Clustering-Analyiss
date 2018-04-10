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


def get_pixel(data):
    pixel = np.zeros((len(data), 784))
    pixel[:] = data[:, 2:]
    return pixel




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




def one_digit(data):
    index = random.sample(list(range(len(data))), 1)
    pixel = get_pixel(data)[index].reshape(28, 28)
    plt.imshow(pixel, cmap='gray')
    plt.show()
    return



def plot_2d(data):
    index = random.sample(list(range(len(data))), 1000)
    data = data[index]
    label = np.zeros((len(data), 1))
    xy = np.zeros((len(data), 2))
    label[:, 0] = data[:, 1]
    xy[:, 0] = data[:, 2]
    xy[:, 1] = data[:, 3]
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(10):
        plt.scatter(xy[label[:, 0] == i, 0], xy[label[:, 0] == i, 1], color=colors[i])
    plt.show()
    return




def diff_k_all(data, k, max_iter):
    wcssd = np.zeros((len(k), 1))
    sc = np.zeros((len(k), 1))
    for i in range(len(k)):
        print k[i]
        index = random.sample(list(range(len(data))), k[i])
        cluster, wcssd[i] = k_mean_cluster(data, index, max_iter)
        sc[i] = silh_coeff(data, cluster)
    print 'wcssd', wcssd
    print 'sc', sc
    plt.plot(k, wcssd, 'r')
    plt.show()
    plt.plot(k, sc, 'blue')
    plt.show()
    return



def digits_data(data, digits):
    new_data = np.empty((0, len(data[0])))
    for i in digits:
        d = data[(data[:, 1]) == str(i)]
        new_data = np.concatenate((new_data, d), axis=0)
    return new_data



def diff_k_part(data, digits, k, max_iter):
    new_data = digits_data(data, digits)
    return diff_k_all(new_data, k, max_iter)


def B3(data, k, max_iter):
    wcssd = np.zeros((len(k), 10))
    sc = np.zeros((len(k), 10))
    for i in range(len(k)):
        print k[i]
        index = random.sample(list(range(len(data))), k[i] * 10)
        for j in range(10):
            index_j = index[j*k[i]:j*k[i]+k[i]]
            cluster, wcssd[i, j] = k_mean_cluster(data, index_j, max_iter)
            sc[i, j] = silh_coeff(data, cluster)
    print 'wcssd', wcssd
    print 'sc', sc
    mean_wcssd = np.mean(wcssd, axis=1)
    var_wcssd = np.var(wcssd, axis=1)
    mean_sc = np.mean(sc, axis=1)
    var_sc = np.var(sc, axis=1)
    plt.errorbar(k, mean_wcssd, color='k', yerr=var_wcssd)
    plt.show()
    plt.errorbar(k, mean_sc, color='r', yerr=var_sc)
    plt.show()
    return



def plot_2d_cluster(data, cluster):
    index = random.sample(list(range(len(data))), 1000)
    data = data[index]
    cluster = cluster[index]
    xy = np.zeros((len(data), 2))
    xy[:, 0] = data[:, 2]
    xy[:, 1] = data[:, 3]
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(len(cluster[0])):
        plt.scatter(xy[cluster[:, i] == 1, 0], xy[cluster[:, i] == 1, 1], color=colors[i])
    plt.show()
    return



def B4(data, k):
    index = random.sample(list(range(len(data))), k)
    cluster, wcssd = k_mean_cluster(data, index, 50)
    NMI = get_NMI(data, cluster)
    plot_2d_cluster(data, cluster)
    return NMI


def sub_data(data):
    sub_data = np.empty((0, 4))
    for i in range(10):
        d = data[(data[:, 1]) == str(i)]
        index = random.sample(list(range(len(d))), 10)
        sub_data = np.concatenate((sub_data, d[index]), axis=0)
    return sub_data


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



def bonus_2(vector):
    vector = np.real(vector)
    for i in range(len(vector[0])):
        print i
        pixel = vector[:, i].reshape(28, 28)
        print pixel
        plt.imshow(pixel, cmap='gray')
        plt.show()
    return


def bonus_3(data, p):
    vector, new_vector = get_PCA(data, p)
    label = np.zeros((len(data), 1))
    label[:, 0] = data[:, 1]
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(10):
        plt.scatter(new_vector[label[:, 0] == i, 0], new_vector[label[:, 0] == i, 1], color=colors[i])
    plt.show()



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





def diff_k_PVA(data, k, p):
    wcssd = np.zeros((len(k), 1))
    sc = np.zeros((len(k), 1))
    for i in range(len(k)):
        print k[i]
        index = random.sample(list(range(len(data))), k[i])
        cluster, wcssd[i] = PCA_kmeans_cluster(data, p, k[i], index)
        sc[i] = silh_PCA(data, cluster)
    print 'wcssd', wcssd
    print 'sc', sc
    plt.plot(k, wcssd, 'r')
    plt.show()
    plt.plot(k, sc, 'blue')
    plt.show()
    return



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






    #data = load_csv('digits-embedding.csv')
    #digits = [2, 4, 6, 7]
    #data = digits_data(data, digits)
    '''
    c = HC_cluster(data, 'single', 32)
    m = trans_HC(data, c, 32)
    print get_NMI(data, m)
    c = HC_cluster(data, 'complete', 32)
    m = trans_HC(data, c, 32)
    print get_NMI(data, m)
    c = HC_cluster(data, 'average', 32)
    m = trans_HC(data, c, 32)
    print get_NMI(data, m)
    digits = [2, 4, 6, 7]
    new_data = digits_data(data, digits)
    print B4(new_data, 4)

    k = 4
    index = random.sample(list(range(len(data))), k)
    cluster, wcssd = k_mean_cluster(data, index, 50)
    NMI = get_NMI(data, cluster)
    print NMI

    cluster, wcssd = k_mean_cluster(data, index, 50)
    print 'WC-SSD', str(wcssd)
    ss = silh_coeff(data, cluster)
    print 'SC', str(ss)

    data = load_csv('digits-raw.csv')
    #digits = [6, 7]
    #data = digits_data(data, digits)
    p = 10
    k = [2, 4, 8, 16, 32]
    #bonus_3(data, p)
    vector, new_vector = get_PCA(data, p)
    bonus_2(vector)
    #diff_k_PVA(new_vector, k, p)

    index = random.sample(list(range(len(data))), 2)
    vector, new_vector = get_PCA(data, p)
    cluster, wcssd = PCA_kmeans_cluster(new_vector, p, 2, index)
    print get_NMI(data, cluster)
    '''

    #bonus_2(vector)
    #plot_2d(data)
    #k = [2, 4, 8, 16, 32]
    #diff_k_all(data, k, 50)
    #digits = [6, 7]
    #new_data = digits_data(data, digits)
    #B3(new_data, k, 50)

    return


main()