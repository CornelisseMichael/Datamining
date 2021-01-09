import os
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn import metrics   

def plot_tree_graph(inp, start, stop):
    dot_data = export_graphviz(inp, out_file=None, 
                               feature_names=range(1,39),  
                               class_names= ['ALL', 'AML'],
                               rounded=True,
                               filled=True)
    # Draw graph
    graph = Source(dot_data, format="png") 
    return graph


def plot_confusion_matrix (cm, label, title = 'Confusion matrix', color = 'Blues'):
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, 
                annot = True, 
                square = True, 
                cmap = color,
                linewidths = 0.5, 
                linecolor = 'Black', 
                cbar = False, 
                xticklabels = label, 
                yticklabels = label)
    plt.title(title)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()
    
    
def plot_roc_curve (true, prediction, model_type):
    FP, TP, TRS = metrics.roc_curve (true, prediction)
    AUC = metrics.roc_auc_score(true, prediction)
    plt.figure(figsize = (5,5))
    plt.plot(FP, TP, label='{} model'.format(model_type), c = 'b')
    plt.plot([0, 1], label='Random guessing', linestyle='dashed', c='k')
    plt.title('ROC curve {}\nAUC of {}' .format(model_type, AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()