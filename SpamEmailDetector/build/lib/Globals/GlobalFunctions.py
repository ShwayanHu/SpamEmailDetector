import numpy as np
from sklearn import metrics


def show_metrics(true_labels,predicted_labels,showMetrics=False):
    accuracy = np.round(metrics.accuracy_score(true_labels,predicted_labels),4)
    precision = np.round(metrics.precision_score(true_labels,predicted_labels),4)
    recall = np.round(metrics.recall_score(true_labels,predicted_labels),4)
    F1 = np.round(metrics.f1_score(true_labels,predicted_labels),4)
    if showMetrics==True:
        print('accuracy:',accuracy)
        print('precision:',precision)
        print('recall:',recall)
        print('F1:',F1)
    return accuracy,precision,recall,F1