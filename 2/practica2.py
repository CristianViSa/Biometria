import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

TRAIN_DATA_PATH = "data/Train/"
TEST_DATA_PATH = "data/Test"

import numpy as np

import matplotlib.pyplot as plt

# Metodo extraido de https://intellipaat.com/community/7530/how-to-read-pgm-p2-image-in-python
def read_pgm(name):
    with open(name) as f:
        lines = f.readlines()
    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
    # here,it makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2'
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
    return (np.array(data[3:]), (data[1], data[0]), data[2])


def read_images(dir):
    images = []
    labels = []
    # Carpetas con el usuario (s1, s2,...)
    for folder in os.listdir(dir):
        try:
            # Los archivos de imagen dentro de las carpetas
            data_path = os.path.join(dir,folder)
            for file in os.listdir(data_path):
                image = read_pgm(os.path.join(data_path, file))
                images.append(image[0])
                size = image[1]
                labels.append(str(folder))
                # plt.imshow(np.reshape(image[0], image[1]))  # Usage example
                # plt.show()
        except:
            print("File not found", folder)
    return np.matrix(images), labels


# data = readpgm(TRAIN_DATA_PATH + "s1/1.pgm")
# print(data)
# plt.imshow(np.reshape(data[0], data[1]))  # Usage example
# plt.show()

read_images(TRAIN_DATA_PATH)
if __name__ == '__main__':
    pass

