import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_loss_accuracy(train_loss, train_acc,
                       validation_loss, validation_acc):
    epochs = len(train_loss)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(list(range(epochs)), train_loss, label='Training Loss')
    ax1.plot(list(range(epochs)), validation_loss, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Epoch vs Loss')
    ax1.legend()

    ax2.plot(list(range(epochs)), train_acc, label='Training Accuracy')
    ax2.plot(list(range(epochs)), validation_acc, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Epoch vs Accuracy')
    ax2.legend()
    fig.set_size_inches(15.5, 5.5)


def plot_confusion_matrix(y_true, y_pred, label_names):
    n_classes = len(label_names)
    confusion_matrix = np.zeros((n_classes, n_classes))
    y_true = [int(y.cpu().numpy()) for y in y_true]
    y_pred = [int(y.cpu().numpy()) for y in y_pred]
    for idx in range(len(y_true)):
        target = y_true[idx]
        output = y_pred[idx]

        confusion_matrix[target][output] += 1

    fig, ax = plt.subplots(1)

    ax.matshow(confusion_matrix, cmap=mpl.colormaps['viridis'])
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)

    ax.set_xlabel('Predictions')
    ax.set_ylabel('Targets')
    ax.set_title('Confusion matrix')

    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="w")

    plt.show()


def plot_dataloader_distribution(dataloader, label_names):
    n_labels = len(label_names)
    batch_distribution = {l: [] for l in range(n_labels)}

    for _, b in dataloader:
        for l in range(n_labels):
            batch_distribution[l].append(int((b == l).sum()))

    x = np.arange(len(batch_distribution[0]))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, freq in batch_distribution.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, freq, width, label=label_names[label])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Batch index')
    ax.set_ylabel('Number of samples')
    ax.set_title('Dataloader class distribution')
    ax.set_xticks(x + width, x)
    ax.legend(loc='upper left')

    plt.show()
