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


def one_digit(data):
    index = random.sample(list(range(len(data))), 1)
    pixel = get_pixel(data)[index].reshape(28, 28)
    plt.imshow(pixel, cmap='gray')
    plt.show()
    return