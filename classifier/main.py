import itertools

import matplotlib
from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from pandas import DataFrame


def NaiveBayes(dataset, show=False):
    gnb = GaussianNB()
    X, labels = dataset

    return gnb.fit(X, labels)


def NaiveBayesTest(dataset, gnb, show=False):
    X, labels = dataset
    new_labels = gnb.predict(X)
    conf = confusion_matrix(labels, new_labels)
    print(conf)
    conf = conf / conf.sum(axis=1)
    plt.matshow(conf)
    plt.title("NaiveBayes")
    plt.show()
    if show:
        for i in range(len(X[0])):
            if i >= 2:
                break
            for j in range(i):
                x = list(x[i] for x in X)
                y = list(x[j] for x in X)

                paint(x, y, labels, 'labels')
                paint(x, y, new_labels, 'prediction')

        plt.show()


def MyBayes(dataset, show=False):
    X, Y = dataset

    X0 = np.array(X[Y <= 0]).astype(int)
    X1 = np.array(X[Y > 0]).astype(int)
    freq0, freq1, ctgr0, ctgr1 = [], [], [], []
    print(X0, X1)
    print(Y)
    for j in range(0, X.shape[1]):
        ct0 = np.unique(X0[:, j])
        ct1 = np.unique(X1[:, j])
        ct0 = np.hstack((ct0, 1000000))
        ct1 = np.hstack((ct1, 1000000))
        fr0 = np.histogram(X0[:, j], bins=ct0)[0] / X0.shape[0]
        fr1 = np.histogram(X1[:, j], bins=ct1)[0] / X1.shape[0]

        freq0.append(fr0)
        freq1.append(fr1)
        ctgr0.append(ct0)
        ctgr1.append(ct1)

    return (freq0, freq1, ctgr0, ctgr1)


def MyBayesTest(dataset, model, show=False):
    X, Y = dataset
    (freq0, freq1, ctgr0, ctgr1) = model
    pred_score_train = np.zeros((X.shape[0], 2))

    for j in range(0, X.shape[1]):
        for n in range(0, pred_score_train.shape[0]):
            idx0 = [i for i, v in enumerate(ctgr0[j]) if (v == int(X[n, j]))]
            idx1 = [i for i, v in enumerate(ctgr1[j]) if (v == int(X[n, j]))]
            print(idx0, idx1)
            print(ctgr0[j], int(X[n, j]))
            p0 = 0.0001
            p1 = 0.0001
            if idx0:
                p0 = freq0[j][idx0][0]
            if idx1:
                p1 = freq1[j][idx1][0]

            pred_score_train[n, 1] += -np.math.log(p0) + np.math.log(p1)

    print("score = ", pred_score_train)
    return pred_score_train


def GaussianClassifier(dataset):
    X, labels = dataset
    kernel = 1.0 * RBF(1.0)
    classifier = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X, labels)
    return classifier


def GaussianClassifierTest(dataset, classifier, show=False):
    X, labels = dataset
    classifier.score(X, labels)
    new_labels = classifier.predict(X)

    conf = confusion_matrix(labels, new_labels)
    print(conf)
    conf = conf / conf.sum(axis=1)

    plt.matshow(conf)
    plt.title("Gaussian")
    plt.show()

    if show:
        for i in range(len(X[0])):
            if i >= 2:
                break
            for j in range(i):
                x = list(x[i] for x in X)
                y = list(x[j] for x in X)

                paint(x, y, labels, "labels")
                paint(x, y, new_labels, "prediction")

        plt.show()


def pca(dataset, show=False):
    X, labels = dataset

    n = X.shape[1]

    pca = PCA(n_components=n)
    pca.fit(X, labels)
    print(pca.score(X, labels))
    df1 = DataFrame(data=X)
    plt.matshow(df1.corr())
    plt.title("Before PCA")
    df2 = DataFrame(data=pca.transform(X))
    plt.matshow(df2.corr())
    plt.title("After PCA")
    plt.show()


def paint(x, y, label, title=None):
    N = max(label) + 1

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title(title)
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, N, N + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax.scatter(x, y, c=label, s=10, cmap=cmap, norm=norm)

    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)


if __name__ == "__main__":
    dataset = load_iris(return_X_y=True)

    x, y = dataset
    y_id = np.nonzero(y < 2)

    x = x[y_id]
    y = y[y_id]
    dataset = x, y
    model = MyBayes(dataset)

    MyBayesTest(dataset, model)
