from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf
import matplotlib.cm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

base_path = Path("cifar100superclass/")

classes = []


def load_classes(path: Path, count=10):
    classes = [y.name for x in path.iterdir() for y in x.iterdir() if x.is_dir() and y.is_dir()]
    classes = classes[-count:]
    print(classes)
    return classes


def load(path: Path):
    x, y, = [], []
    print(classes)
    for v1 in path.iterdir():
        for v2 in v1.iterdir():
            if not v2.name in classes:
                continue
            print("reading class {}".format(v2.name))
            for pic in v2.iterdir():
                #     print("reading picture {}".format(pic))
                img = cv2.imread(str(pic))
                img_size = img.shape[0]
                x.append(img)
                y.append(v2.name)

    x = np.asarray(x).reshape(-1, img_size, img_size, 3) / 255
    y = [classes.index(x) for x in y]
    y = np.array(y)

    return x, y


def make_model(shape):
    model = Sequential()
    print(shape)
    model.add(Conv2D(16, (3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(classes)))
    model.add(Activation('softmax'))
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


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

    print(cm)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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


def get_out(model, x):
    y = model.predict(x)
    y = np.asarray([list(x).index(np.max(x)) for x in y])
    return y


if __name__ == "__main__":
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    classes = load_classes(base_path.joinpath("train"), 10)

    x, y = load(base_path.joinpath("train"))

    X, Y = load(base_path.joinpath("test"))
    model = make_model(x.shape[1:])
    print(model.summary())
    model.fit(x, y, epochs=50)

    Y_pred = get_out(model, X)
    y_pred = get_out(model, x)
    plot_confusion_matrix(Y, Y_pred, np.asarray(classes), normalize=True, title="test")
    plot_confusion_matrix(y, y_pred, np.asarray(classes), normalize=True, title="train")
    plt.show()
