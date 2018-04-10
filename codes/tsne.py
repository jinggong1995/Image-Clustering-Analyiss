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



def main():

    data = load_csv('digits-embedding.csv')
    plot_2d(data)


main()