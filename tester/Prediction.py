import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import helper
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import os
from sklearn.utils.multiclass import unique_labels

import random

TEST_IMAGES_DIR = 'D:/Downloads/manwomandataset/dataset/test/TestImages'
LABELS_FILE = 'D:/Downloads/manwomandataset/labels.txt'
IMG_ROWS = 96
IMG_COLUMNS = 96
NB_CHANNEL = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLUMNS, NB_CHANNEL)
MODEL_DIR = 'D:/Downloads/manwomandataset/Models/ManWomanVgg42/Last/'
MODEL_NAME = 'last-vgg3-model-3396-100e-25b-96s.h5'
PLOT_IMG = False


def get_test_img():
    image_set = glob.glob(TEST_IMAGES_DIR + "/*.jpg")
    images = []
    labels = []
    for img in image_set:
        tmp_img = Image.open(img)
        images.append(np.array(helper.resize_ratio(tmp_img, IMG_ROWS)))
        label = np.zeros(num_classes)
        label = define_label(img, label)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def define_label(img, label):
    for c in class_names:
        if c in os.path.basename(img):
            label[class_names.index(c)] = 1
    return label


def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    print("predicted label", predicted_label)
    color = 'blue'
    plt.xlabel("{}".format(class_names[int(predicted_label)], color=color))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

class_names = open(LABELS_FILE).read().split(",")
print(class_names)
num_classes = len(class_names)
img_test, labels = get_test_img()
model = keras.models.load_model(MODEL_DIR + '/' + MODEL_NAME)
test_lost, test_acc = model.evaluate(img_test, labels)
predictions = model.predict(img_test)
print("prediction", predictions)
print("accuracy = ", test_acc)
print("loss = ", test_lost)
plot_confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1), classes=class_names,
                      title='Confusion matrix')
plot_confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1), classes=class_names, normalize=True,
                      title='Confusion matrix Normalized')
plt.show()

if PLOT_IMG:
    predictions = model.predict(img_test)
    for i in range(img_test.shape[0]):
        plot_image(i, predictions, img_test)