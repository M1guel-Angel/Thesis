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


def box_plost(minimum, maximum, average, names, title='Boxes plots', save_name='box_plot'):
    data = []
    for i in range(len(minimum)):
        data.append(np.concatenate(([minimum[i]], [maximum[i]], [average[i]]), 0))

    fig, axs = plt.subplots()

    # # basic plot
    boxprops = {'color': 'darkorange', 'linestyle': '-', 'linewidth': 0.7}
    medianprops = {'color': 'darkred', 'linestyle': '-', 'linewidth': 0.7}
    capprops = {'color': 'darkred', 'linestyle': '-', 'linewidth': 0.6}
    axs.boxplot(data, boxprops=boxprops, medianprops=medianprops, capprops=capprops)

    pos = np.arange(len(names)) + 1
    upperLabels = [str((int(np.round(minimum[s])), int(np.round(average[s])), int(np.round(maximum[s]))))
                   for s in range(len(minimum))]
    down = 50
    for tick, label in zip(range(len(names)), axs.get_xticklabels()):
        axs.text(pos[tick], down + down*0.05, upperLabels[tick],
                 horizontalalignment='center', size='x-small', weight='bold',
                 color='darkred')

    axs.set_title(title)
    axs.set_ylim(down, 100)
    axs.set_xticklabels(names, rotation=15, fontsize=8)

    fig.savefig(save_name)
    plt.close(fig)


if __name__ == '__main__':
    names = ['spec_train', 'spec_test', 'rp_spec_train', 'rp_spec_val']
    minim = [73.6842107504, 66.6666686535, 91.729323129, 78.9473681073]
    maxim = [96.2406017278, 91.2280706983, 100, 98.2456150808]
    avers = [86.1904756691, 77.8947370094, 98.092, 92.2753333]

    box_plost(minim, maxim, avers, names, title='Precisión con el uso de espectrogramas', save_name='accuracy_spectrogram_plot')
