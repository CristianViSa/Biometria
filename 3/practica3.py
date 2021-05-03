import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg as LA
import scipy.linalg as scipylg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
TRAIN_DATA_PATH = "data/Train/"
TEST_DATA_PATH = "data/Test"

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


def calculateLDA(matrix, labels):
    n_features = matrix.shape[0]
    class_labels = np.unique(labels)

    MU = matrix.mean(axis = 1)
    SB = np.zeros((n_features, n_features))
    SW = SB.copy()

    for label in class_labels:
        classes = np.where(label == labels)
        matrix_class = matrix[:, classes[0]]

        MU_class = matrix_class.mean(axis = 1)
        class_samples = matrix_class.shape[1]

        SB += class_samples *  (MU_class - MU) *  (MU_class - MU).T
        SW += (matrix_class - MU_class) * (matrix_class - MU_class).T

    try:
        SWi = LA.inv(SW).dot(SB)
        D, B = np.linalg.eig(SWi)
        D, B = scipylg.eig(SB, SW)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        sort_indexes = D.argsort()[::-1]
        D = D[sort_indexes]
        B = B[:, sort_indexes]

    except:
        print("SW has no inverse")

    return B

def transform(images, matrix, dim):
    eigenvectors = matrix[:, 1 : dim + 1]
    mean = images.mean(axis = 0)
    transformed_images = (images-mean) * eigenvectors

    return transformed_images


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
    sort_indexes = D_p.argsort()[::-1]
    D_p = D_p[sort_indexes]
    B_p = B_p[:, sort_indexes]
    B = A * B_p
    D = float(d/n) * D_p
    # print(D)
    return B/LA.norm(B, axis = 0)


if __name__ == '__main__':
    train_images, train_labels = read_images(TRAIN_DATA_PATH)
    test_images, test_labels = read_images(TEST_DATA_PATH)

    PCA_matrix = calculate_PCA(train_images)
    train_reduced_pca = transform(train_images, PCA_matrix, 200)
    test_reduced_pca = transform(test_images, PCA_matrix, 200)

    LDA_matrix = calculateLDA(train_reduced_pca.T, np.array(train_labels))
    scores = []
    for dim in range(1, 39):
        train_reduced_LDA = transform(train_reduced_pca, LDA_matrix, dim)
        test_reduced_LDA = transform(test_reduced_pca, LDA_matrix, dim)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_reduced_LDA, train_labels)
        scores.append(clf.score(test_reduced_LDA, test_labels))
    scores = np.array(scores)
    print("Best result ", np.amax(scores), "with ", np.argmax(scores), " components")
    plt.plot(scores)
    plt.xlabel("dimensions")
    plt.ylabel("accuracy")
    plt.show()