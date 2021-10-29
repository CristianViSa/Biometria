import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg as LA

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
    return np.matrix(images), labels

def calculate_PCA(images):
    images = images.T

    d = images.shape[0]
    n = images.shape[1]

    mean = images.mean(axis=1)
    A = images - mean
    C = float(1/d) * A.T * A

    D_p, B_p = LA.eig(C)
    sort_indexes = D_p.argsort()[::-1]
    D_p = D_p[sort_indexes]
    B_p = B_p[:, sort_indexes]
    B = A * B_p
    D = float(d/n) * D_p

    return B/LA.norm(B, axis = 0)

def transform(images, matrix, dim):
    eigenvectors = matrix[:, 1 : dim + 1]
    mean = images.mean(axis = 0)
    transformed_images = (images-mean) * eigenvectors

    return transformed_images

if __name__ == '__main__':
    train_images, train_labels = read_images(TRAIN_DATA_PATH)
    test_images, test_labels = read_images(TEST_DATA_PATH)
    matrix = calculate_PCA(train_images)

    # plt.imshow(np.reshape(matrix[:, 2], (112, 92)))
    # plt.show()

    train = transform(train_images, matrix, 30)
    test = transform(test_images, matrix, 30)

    clf = KNeighborsClassifier(n_neighbors = 1)
    clf.fit(train, train_labels)
    print("Accuracy with 30 dimensions : ", clf.score(test, test_labels))

    scores = []
    matrix = calculate_PCA(train_images)
    for dim in range(1, 200):
        train = transform(train_images, matrix, dim)
        test = transform(test_images, matrix, dim)
        clf = KNeighborsClassifier(n_neighbors = 1)
        clf.fit(train, train_labels)
        scores.append(clf.score(test, test_labels))
    scores = np.array(scores)
    print("Best result ", np.amax(scores), "with " , np.argmax(scores), " components")
    fig, ax = plt.subplots()
    ax.plot(scores)
    plt.xlabel("dimensions")
    plt.ylabel("accuracy")
    plt.savefig("pca.png")
    plt.show()