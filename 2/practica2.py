import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg as LA

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

def calculate_PCA(images):
    images = images.T
    d = images.shape[0]
    n = images.shape[1]
    # print(images.shape)
    # print(d, n)
    # print("d = ", d , "n = ", n)
    mean = images.mean(axis=1)
    A = images - mean
    # print("A : " ,A.shape)
    C = float(1/d) * A.T * A

    D_p, B_p = LA.eig(C)
    # print("C : " ,C.shape)
    # print("D_p : " ,D_p.shape)
    # print("B_p : " ,B_p.shape)
    # Ordenar los eigenvalues y eigenvectors
    sort_indices = D_p.argsort()[::-1]
    D_p = D_p[sort_indices]
    B_p = B_p[:, sort_indices]
    B = A * B_p
    D = float(d/n) * D_p
    # print(D)
    return B/LA.norm(B, axis = 0)

def transform(images, matrix, dimensions):
    eigenvectors = matrix[:, 1:dimensions+1]
    mean = images.mean(axis = 0)
    transformed_images = (images-mean) * eigenvectors

    return transformed_images

if __name__ == '__main__':
    train_images, train_labels = read_images(TRAIN_DATA_PATH)
    test_images, test_labels = read_images(TEST_DATA_PATH)
    matrix = calculate_PCA(train_images)
    plt.imshow(np.reshape(matrix[:, 7], (112, 92)))
    plt.show()

    train = transform(train_images, matrix, 179)
    test = transform(test_images, matrix, 179)

    clf = KNeighborsClassifier(n_neighbors = 1)
    clf.fit(train, train_labels)
    print(clf.score(test, test_labels))

    scores = []
    matrix = calculate_PCA(train_images)
    for dim in range(1, 200):
        train = transform(train_images, matrix, dim)
        test = transform(test_images, matrix, dim)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train, train_labels)
        scores.append(clf.score(test, test_labels))
    scores = np.array(scores)
    print("Best result ", np.amax(scores), "with " , np.argmax(scores), " components")

    #pca = PCA().fit(train_images)

    # Ver matriz con la varianza
    # plt.figure(figsize=(18, 7))
    # plt.plot(pca.explained_variance_ratio_.cumsum())
    # print(np.where(pca.explained_variance_ratio_.cumsum() > 0.95))
    # plt.show()
    # Comprobar donde la energia (varianza) es mayor de un 95
    # PCA CON SKLEARN
    # results = []
    # variance_sum = []
    # for i in range(1, 100):
    #     pca = PCA(n_components=i)
    #     tr_img = pca.fit_transform(train_images)
    #
    #     variance_sum = pca.explained_variance_ratio_.cumsum()
    #     tst_img = pca.transform(test_images)
    #     clf = KNeighborsClassifier(n_neighbors = 1)
    #     clf.fit(tr_img, train_labels)
    #     results.append(clf.score(tst_img, test_labels))
    #
    # print(results)
    # print(np.where(variance_sum > 0.95))
    # plt.plot(list(range(1, 200)), results)
    # plt.xlabel("dimensiones")
    # plt.ylabel("accuracy")
    # plt.plot()
    # results = np.array(results)
    # print("Best result ", np.amax(results), "with " , np.argmax(results), " components")
    # n = 200
    # d = 100x100 casi
    # Truco --> No calcular diagonalizacion Cdxd se calcula C`nxn
    # Ojo, cuando n es mucho mas pequeño que d
    # Despues, hay que hacer dichos vectores ortonormales
