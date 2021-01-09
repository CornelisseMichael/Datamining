import os
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn import metrics   


def get_tree_graph(inp, start, stop, labels):
    """
    Retruns a graph for the tree

    Parameters
    ----------
    inp : str
        input data for the tree we want to graph
    start : int
        the start index for the feuture_names
    stop : int
        the stop index for the feuture_names
    labels : int
        the labels for the class_names
        """
    dot_data = export_graphviz(
        inp,
        out_file=None,
        feature_names=range(start,stop),
        class_names= labels,
        rounded=True,
        filled=True)
    # Draw graph
    graph = Source(dot_data, format="png") 
    return graph

def plot_cm_and_rc(cm, label, y_test, prediction, method, title_cm='Confusion matrix'):
    """
    Plots confusion matrix and roc_curve

    Parameters
    ----------
    cm : confusion matrix
        confusion matrix data
    label : int
        labels for plot
    y_test : 
        y_test data
    method : str
        type of algorithm thats being displayed e.g. tree, rf, xgb
    title_cm : str
        title for confusuion matrix
        """
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    sns.heatmap(cm, 
                annot = True, 
                square = True, 
                cmap = 'Blues',
                linewidths = 0.5, 
                linecolor = 'Black', 
                cbar = False,
                xticklabels = label, 
                yticklabels = label)
    plt.title(title_cm)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    
    #ROC curve and Area under the curve plotting
    FP, TP, TRS = metrics.roc_curve (y_test, prediction)
    AUC = metrics.roc_auc_score(y_test, prediction)
    plt.subplot(222)
    plt.plot(FP, TP, label='{} model'.format(method), c = 'b')
    plt.plot([0, 1], label='Random guessing', linestyle='dashed', c='k')
    plt.title('ROC curve {}\nAUC of {}' .format(method, AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()