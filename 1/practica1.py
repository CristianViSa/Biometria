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


def FnFp(fpr, fnr, thresholds):
    best_dif= 1
    index = 0
    for i in range(len(thresholds)):
        fp = fpr[i]
        fn = fnr[i]
        th = thresholds[i]
        if fp == fn:
            return fp, fn, th
        else:
            dif = abs(fp-fn)
            if best_dif > dif:
                best_dif = dif
                index = i
    return fpr[index], fnr[index], thresholds[index]

def FNatFP(fpr, fnr, thresholds, x):
    for i in range(len(thresholds)):
        fp = fpr[i]
        fn = fnr[i]
        th = thresholds[i]
        if fp <= x:
            return fp, fn, th
    return None, None, None


def FPatFN(fnr, fpr,  thresholds, x):
    for i in range(len(thresholds)):
        fp = fpr[i]
        fn = fnr[i]
        th = thresholds[i]
        if fn <= x:
            return fp, fn, th
    return None, None, None

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
    fnr = []
    tnr = []
    thresholds = []
    for val in values:
        thresholds.append(val)
    for th in thresholds:
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        for i in range(len(values)):
            if values[i] > th:
                if labels[i] == 1:
                    tp = tp + 1
                elif labels[i] == 0:
                    fp = fp + 1
            else:
                if labels[i] == 1:
                    fn = fn + 1
                elif labels[i] == 0:
                    tn = tn + 1
        fpr.append(fp / negatives)
        tpr.append(tp / positives)
        fnr.append(fn / positives)
        tnr.append(tn / negatives)
    if plot:
        plt.plot(fpr, tpr)
        plt.title("ROC")
        plt.axis([0, 1, 0, 1])
        plt.savefig("aoc.png")
        plt.show()
    auc(values, labels)
    return fpr, tpr, fnr, tnr, thresholds


if __name__ == '__main__':
    #args = parse_args()
    scores = read_files(DATA_PATH + "scoresA_clientes", DATA_PATH + "scoresA_impostores")
    fpr, tpr ,fnr, tnr, thresholds = roc(scores, plot=False)
    fp, fn, th = FNatFP(fpr, fnr, thresholds, 0.2)
    fp, fn, th = FPatFN(fpr, fnr, thresholds, 0.2)
    fp, fn, th = FnFp(fpr, fnr, thresholds)
    d_prime(scores)

