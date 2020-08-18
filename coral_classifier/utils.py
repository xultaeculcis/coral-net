import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from coral_classifier.dataset import CoralFragDataset


#  --- Utility functions ---

def print_system_info() -> None:
    # If there's a GPU available...
    if torch.cuda.is_available():
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the:', torch.cuda.get_device_name(0))
        print('GPU capability:', torch.cuda.get_device_capability(0))
        print('GPU properties:', torch.cuda.get_device_properties(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')


def _plot_confusion_matrix(cm,
                           target_names,
                           title='Confusion matrix',
                           cmap=None,
                           normalize=True):
    """
    Given a confusion matrix (cm), make a nice plot.
    Based on Scikit Learn's implementation.

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
    """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    figure = plt.figure(figsize=(20, 16))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def plot_single_batch(loader: DataLoader, dataset: CoralFragDataset) -> None:
    batch = next(iter(loader))
    fig, ax = plt.subplots(8, 4, figsize=(12, 20))
    idx = 0
    for i in range(8):
        for j in range(4):
            image = batch[0][idx].permute(1, 2, 0)
            target = batch[1][idx].item()
            ax[i, j].set_title(dataset.class_lookup_by_index[target])
            ax[i, j].imshow(image)
            idx += 1

    plt.show()
