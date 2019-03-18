# -*- coding: utf-8 -*-


import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels


labels = ['bus', 'train', 'motorcycle']
labels = {label: i for i, label in enumerate(labels)}
img_size = 16
num_classes = len(labels)


def load_data(folder_path):
    X = []
    y = []
    for root, dirs, files in os.walk(folder_path):
        for basename in files:
            if basename.endswith(".png"):
                file_path = os.path.join(root, basename)
                _, label = os.path.split(root)
                label = labels.get(label)
                if label is None:
                    continue
                img = Image.open(file_path)
                img.load()
                img.thumbnail((img_size, img_size))
                img = np.asarray(img, dtype=np.int16)
                X.append(img)
                y.append(label)
    X = np.asarray(X).reshape(-1, img_size, img_size, 3) / 255
    y = np.asarray(y).reshape(-1, 1)
    return X, y


X, y = load_data("/home/discovery/Work/cnn/cifar100superclass/train")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

X_test, y_test = load_data("/home/discovery/Work/cnn/cifar100superclass/test")

model = Sequential()
model.add(Conv2D(16, (3, 3),input_shape=X_train.shape[1:]))
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
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val), shuffle=True)
#print(model.summary())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


y_pred = model.predict_classes(X_test)

class_names = list(labels.keys())
print(classification_report(y_test, y_pred, target_names=class_names))

for i in np.random.choice(len(X_test), 5):
    img = (X_test[i].reshape(img_size, img_size, 3)*255).astype(int)
    plt.imshow(img)
    plt.xlabel("{} -> {}".format(class_names[int(y_test[i])],
               class_names[int(y_pred[i])]))
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


# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=np.array(class_names), title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=np.array(class_names), normalize=True, title='Normalized confusion matrix')

plt.show()
