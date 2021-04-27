import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import roc_auc_score


DATA_PATH = "data/"


def read_file(filename, label):
    file = open(filename, 'r')
    # Formato : Cliente Score
    scores = []
    for line in file:
        client, score = line.split()
        scores.append([score, label])
    return scores


def auc():
    1==1

def roc(scores, positives, negatives, plot=True):
    values = [val[0] for val in scores]
    labels = [lab[1] for lab in scores]
    test = roc_auc_score(labels, values)
    n1 = positives
    n0 = len(scores) - n1
    print("AUC : ")
    print(test)
    print(1 - 0.8831634391249776)
    fpr = []
    tpr = []
    thresholds = []
    for val in values:
        if val not in thresholds:
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
    values1 = [val[0] for val in scores_c]
    values2 = [val[0] for val in scores_i]
    u_statistic, pVal = stats.mannwhitneyu(values1, values2)
    multi = len(values1) * len(values2)
    # Print results

    print('P value:')
    print(pVal)
    print('EL AUC QUE PARECE SER')
    print(u_statistic /multi)
    scores = scores_i + scores_c
    scores = sorted(scores)
    roc(scores, len(scores_c), len(scores_i))

