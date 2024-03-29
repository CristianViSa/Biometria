import math

import matplotlib.pyplot as plt
import numpy as np
import argparse

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
    dif = np.abs(fnr-fpr)
    arg_min_dif = np.argmin(dif)

    th = thresholds[arg_min_dif]
    fp = fpr[arg_min_dif]
    fn = fnr[arg_min_dif]

    return fp, fn, th

def FNatFP(fpr, fnr, thresholds, x):
    dif = np.abs(fpr-x)
    minindex = np.where(dif == dif.min())
    minfn = 1
    finalindex = 0
    for index in minindex[0]:
        fn = fnr[index]
        if(fn < minfn):
            finalindex = index
            minfn = fn
    th = thresholds[finalindex]
    fp = fpr[finalindex]
    fn = fnr[finalindex]
    return fp, fn, th
def FPatFN(fpr, fnr,  thresholds, x):
    dif = np.abs(fnr-x)
    minindex = np.where(dif == dif.min())
    minfp = 1
    finalindex = 0
    for index in minindex[0]:
        fp = fpr[index]
        if(fp < minfp):
            finalindex = index
            minfp = fp
    th = thresholds[finalindex]
    fp = fpr[finalindex]
    fn = fnr[finalindex]
    return fp, fn, th

def d_prime(scores):
    impostor_scores = np.array([float(val[0]) for val in scores if val[1] == 0])
    client_scores = np.array([float(val[0]) for val in scores if val[1] == 1])
    d_prime = abs(impostor_scores.mean() - client_scores.mean()) / (math.sqrt(impostor_scores.var() + client_scores.var()))

    return d_prime

def auc(scores):
    values = np.array([float(val[0]) for val in scores])
    labels = np.array([int(lab[1]) for lab in scores])
    order = np.argsort(values)
    rank = np.argsort(order)
    rank += 1
    positives = np.sum(labels==1)
    negatives = len(labels) - positives
    U1 = np.sum(rank[labels == 1]) - positives * (positives + 1) / 2

    AUC1 = U1 / (positives * negatives)

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
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_ylabel('1 - FNR')
        ax.set_xlabel('FPR')
        plt.title("ROC")
        plt.axis([0, 1, 0, 1])
        plt.savefig("roc.png")
        plt.show()
    return np.array(fpr), np.array(tpr), np.array(fnr), np.array(tnr), np.array(thresholds)


if __name__ == '__main__':
    # ArgPaser
    parser = argparse.ArgumentParser(description='Practica 1 Biometria')
    parser.add_argument('-x', '--x', type=float, default=0.05,
                        help='Valor de x para calcular FP(FN=x), FN(FP=x)')
    parser.add_argument('-c', '--clients', type=str, default='scoresB_clientes',
                        help='nombre del fichero con datos de clientes')
    parser.add_argument('-i', '--impostors', type=str, default='scoresB_impostores',
                        help='nombre del fichero con datos de impostores')

    args = parser.parse_args()
    x = args.x
    clients_filename = args.clients
    impostors_filename = args.impostors

    scores = read_files(DATA_PATH + clients_filename, DATA_PATH + impostors_filename)
    # Curva ROC, guarda un fichero png (roc.png)
    fpr, tpr ,fnr, tnr, thresholds = roc(scores, plot=True)
    # FP (FN = X) y umbral
    print("FP cuando FN = ", x)
    fp, fn, th = FPatFN(fpr, fnr, thresholds, x)
    print("FP = ", fp)
    print("FN = ", fn)
    print("Umbral = ", th)
    print("------------------------------")

    # FN (FP = X) y umbral
    print("FN cuando FP = ", x)
    fp, fn, th = FNatFP(fpr, fnr, thresholds, x)
    print("FP = ", fp)
    print("FN = ", fn)
    print("Umbral = ", th)
    print("------------------------------")

    # FP = FN y umbral
    print("Umbral cuando FN = FP ")
    fp, fn, th = FnFp(fpr, fnr, thresholds)
    print("FP = ", fp)
    print("FN = ", fn)
    print("Umbral = ", th)
    print("------------------------------")

    # Area bajo la curva ROC
    print("Area bajo la curva ROC = " , auc(scores))
    print("------------------------------")

    # D-Prime
    print("Valor D-Prime = ", d_prime(scores))

