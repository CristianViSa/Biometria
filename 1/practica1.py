import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


DATA_PATH = "data/"


def read_files(clients_filename, impostors_filename):
    clients_file = open(clients_filename, 'r')
    impostors_file = open(impostors_filename, 'r')
    # Formato : Cliente Score
    scores = []
    for line in clients_file:
        client, score = line.split()
        scores.append((float(score), 1))

    for line in impostors_file:
        client, score = line.split()
        scores.append((float(score), 0))
    return np.array(sorted(scores))

def d_prime(scores):
    impostor_scores = np.array([float(val[0]) for val in scores if val[1] == 0])
    client_scores = np.array([float(val[0]) for val in scores if val[1] == 1])
    d_prime = abs(impostor_scores.mean() - client_scores.mean()) / (math.sqrt(impostor_scores.var() + client_scores.var()))

    return d_prime

def auc(values, labels):
    order = np.argsort(values)
    rank = np.argsort(order)
    rank += 1
    positives = np.sum(labels==1)
    negatives = len(labels) - positives
    U1 = np.sum(rank[labels == 1]) - positives * (positives + 1) / 2
    U0 = np.sum(rank[labels == 0]) - negatives * (negatives + 1) / 2

    AUC1 = U1 / (positives * negatives)
    AUC0 = U0 / (positives * negatives)

    print(AUC1)
    return AUC1

def roc(scores, plot=True):

    values = np.array([float(val[0]) for val in scores])
    labels = np.array([int(lab[1]) for lab in scores])
    positives = np.sum(labels==1)
    negatives = len(labels) - positives

    fpr = []
    tpr = []
    thresholds = []
    for val in values:
        thresholds.append(val)
    for th in thresholds:
        fp = 0
        tp = 0
        for i in range(len(values)):
            if values[i] > th:
                if labels[i] == 1:
                    tp = tp + 1
                elif labels[i] == 0:
                    fp = fp + 1
        fpr.append(fp / negatives)
        tpr.append(tp / positives)
    if plot:
        plt.plot(fpr, tpr)
        plt.title("ROC")
        plt.axis([0, 1, 0, 1])
        plt.savefig("aoc.png")
        plt.show()
    auc(values, labels)
    return fpr, tpr
"""
    Columna de clientes, desechar
    3 parametros, fichero clientes, fichero impostores y  X
    PDF con memoria y algunos resultados
    PARA calcular auc --> mann whitney stadistic
     - buen punto FN = FP
     - FP(FN = X) --> Alto confort, acepta clientes facilmente X = 1% , ¿que umbral provee? el que mas se acerce a dicho valor
     - FN(FP = X) --> Alta seguridad, no acepta clientes facilmente X = 
     - Si hay mas de un umbral que da el mismo umero de FN o FP, elegir el umbral que minimize la otra metrica de error
    D prime es proporcional a la resta de las medias  e inversamente proporcional a las varianzas
"""

if __name__ == '__main__':
    scores = read_files(DATA_PATH + "scoresA_clientes", DATA_PATH + "scoresA_impostores")
    # Print results
    # print('P value:')
    # print(pVal)
    # print('EL AUC QUE PARECE SER')
    # print(u_statistic /multi)
    fpr, tpr = roc(scores, plot=False)
    d_prime(scores)
    #print(len(fpr))

