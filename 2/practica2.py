import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

TRAIN_DATA_PATH = "data/Train/"
TEST_DATA_PATH = "data/Test"

import numpy as np

import matplotlib.pyplot as plt

def plot_faces(image):
    plt.imshow(np.reshape(image[0], image[1]))  # Usage example
    plt.show()

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
                #plot_faces(image)
        except:
            print("File not found", folder)
    print("Imagenes procesadas %d total de individuos %d tamañoImagen %s " % (len(images), len(set(labels)), size))
    return np.matrix(images), labels



if __name__ == '__main__':
    train_images, train_labels = read_images(TRAIN_DATA_PATH)
    test_images, test_labels = read_images(TEST_DATA_PATH)

