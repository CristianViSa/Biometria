import numpy as np
import matplotlib.pyplot as plt
DATA_PATH = "data/"

def read_file(filename, label):
    file = open(filename, 'r')
    # Formato : Cliente Score
    scores = []
    for line in file:
        client, score = line.split()
        scores.append([score, label])
    return scores

def roc(scores):
    values = [val[0] for val in scores]
    labels = [lab[1] for lab in scores]

    fpr = []
    tpr = []
    thresholds = []
    for val in values:
        if val not in thresholds:
            thresholds.append(val)
    print(thresholds)
    print(values)
    print(labels)
    p = 0
    n = 0
    for lab in labels:
        if lab == 0:
            n += 1
        else:
            p += 1
    print(p)

    for th in thresholds:
        FP = 0
        TP = 0
        for i in range(len(values)):
            if (values[i] > th):
                if labels[i] == 1:
                    TP = TP + 1
                elif labels[i] == 0:
                    FP = FP + 1
        fpr.append(FP/n)
        tpr.append(TP/p)
    plt.scatter(fpr, tpr)
    plt.show()
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
    scores_c = read_file(DATA_PATH + "scoresA_clientes", 1)
    scores_i = read_file(DATA_PATH + "scoresA_impostores", 0)
    scores = scores_i + scores_c
    scores = sorted(scores)
    roc(scores)

