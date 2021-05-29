from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def confusionMatrix(y_test, y_pred):
    # Print the confusion matrix
    print(metrics.confusion_matrix(y_test, y_pred))

    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_test, y_pred, digits=3))

def rocAUC(x_train, y_train, x_test, y_test, all_clf, clf_labels, colors, linestyles):
    for clf, label, clr, ls \
            in zip(all_clf,
                   clf_labels, colors, linestyles):
        # assuming the label of the positive class is 1
        y_pred = clf.fit(x_train,
                         y_train).predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                         y_score=y_pred,
                                         pos_label=3)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr,
                 color=clr,
                 linestyle=ls,
                 label='%s (auc = %0.2f)' % (label, roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],
             linestyle='--',
             color='gray',
             linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(alpha=0.5)
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')

    #fig = plt.figure()
    #plt.plot(range(10))
    #fig.savefig('../data_base/images/temp.png', dpi=fig.dpi)
    #plt.savefig('../data_base/images', dpi=300)
    plt.show()