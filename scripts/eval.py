import torch
from utils.settings_class import settings
from utils.loaders import getloaders

def get_classes_accuracy(model, data_loader, classes):
    n = len(classes)
    classes = [i for i in range(n)]
    correct = [0 for i in range(n)]
    total = [0 for i in range(n)]

    for imgs, labels in data_loader:
        
        #############################################
        #To Enable GPU Usage
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################
        
        output = model(imgs)
        preds = output.max(1)[1]
        for c in classes:
            correct[c] += ((preds == labels) * (labels == c)).float().sum().item()
            total[c] += max((labels == c).sum().item(), 1)

    return [i / j for i, j in zip(correct, total)]

def get_confusion_matrix(model, data_loader, classes):
    n = len(classes)
    confusion_matrix = torch.zeros(n, n, dtype=torch.int64)

    with torch.no_grad():
        for imgs, labels in data_loader:
        
            #############################################
            #To Enable GPU Usage
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################
            
            output = model(imgs)
            preds = output.max(1)[1]

            for p, t in zip(preds.view(-1), labels.view(-1)):
                confusion_matrix[p.long(), t.long()] += 1
    return confusion_matrix


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

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
    plt.show()

def get_precision(confusion_matrix):
    return confusion_matrix.diag() / confusion_matrix.sum(1)

def get_recall(confusion_matrix):
    return confusion_matrix.diag() / confusion_matrix.sum(0)

def evaluate_model(weight_path, settings):
    train_loader, val_loader, test_loader = getloaders(run_settings)



if __name__ == '__main__':
    weights = 
    eval_settings = settings()
    eval_settings.
    evaluate_model(weights, eval_settings)
