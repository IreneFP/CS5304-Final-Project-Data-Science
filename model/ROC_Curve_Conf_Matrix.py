# This is the entire list of dependencies you need
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix as confmat
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import itertools
import matplotlib.pyplot as plt
from plot_metric.functions import BinaryClassification

# The below part should be pasted under Dev Performance
# dev[0]
"""
ys, y_stars = net.get_eval_data(dev)
cm = confmat(ys, y_stars, normalize='true')
print(cm)
target_names = [0, 1]
print(f1(ys, y_stars))
"""
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.cm.Blues

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.grid(False)
    plt.show()

# Plot the confusion matrix
# plot_confusion_matrix(cm, target_names)


# # Two ways of plotting the ROC Curve
"""
# # ROC Curve - Style 1
fpr, tpr, _ = roc_curve(ys, y_stars)
auc = roc_auc_score(ys, y_stars)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label="ROC Curve, area="+str(auc))
plt.plot([0, 1], [0, 1], lw = 2, linestyle='--', label="Random guess")
plt.legend(loc='lower right')
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# ROC Curve - Style 2 (I think we chose this in the report)
# ROC Curve Visualisation with plot_metric
bc = BinaryClassification(ys, y_stars, labels=[0, 1])

# # Figures
plt.figure(figsize=(8,6))
bc.plot_roc_curve()
plt.show()
"""