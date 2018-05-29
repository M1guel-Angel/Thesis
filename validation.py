import itertools
import numpy as np
import matplotlib.pyplot as plt
from utils import ITER


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues, save_plot=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0, len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment = "center",
        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_plot:
        figure.savefig("confusion_matrix_{0}.jpg".format(ITER[0]))
    # plt.show()
    print("Figure saved")
    plt.close(figure)


def plot_history(history, save_plot=True):
    # summarize history for accuracy
    figure = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val_test'], loc='upper left')
    if save_plot:
        figure.savefig("history_accuracy_{0}.jpg".format(ITER[0]))
    # plt.show()
    plt.close(figure)

    # summarize history for loss
    figure = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val_test'], loc='upper left')
    if save_plot:
        figure.savefig("history_loss_{0}.jpg".format(ITER[0]))
    # plt.show()
    print("Figure saved")
    plt.close(figure)
