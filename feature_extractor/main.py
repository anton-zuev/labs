from keras.applications import NASNetMobile
from keras_applications.resnet_common import ResNet152
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.applications import MobileNet

from util import *

from keras.applications.vgg16 import preprocess_input


def pca(X, labels, new_size):
    n = X.shape[1]

    pca = PCA(n_components=new_size)
    pca.fit(X, labels)
    print(pca.score(X, labels))
    #  df1 = DataFrame(data=X)
    # plt.matshow(df1.corr())
    #  plt.title("Before PCA")
    # df2 = DataFrame(data=pca.transform(X))
    # plt.matshow(df2.corr())
    # plt.title("After PCA")
    # plt.show()
    return pca.transform(X)


def load(path: Path, classes, size, flat=True):
    y_real = []
    imgs = []
    for v1 in path.iterdir():
        for v2 in v1.iterdir():
            if not v2.name in classes:
                continue
            print("reading class {}".format(v2.name))
            c = 0
            for pic in v2.iterdir():
                img = image.load_img(str(pic), target_size=size)
                if flat:
                    img_data = image.img_to_array(img)
                    img_data = preprocess_input(img_data)
                    imgs.append(img_data)
                else:
                    imgs.append(np.asarray(img))
                y_real.append(classes.index(v2.name))

                c += 1
                # if c > 100:
                #    break

    vgg16_feature_list_np = np.array(imgs)
    return vgg16_feature_list_np, y_real


def clusterize(x, y, classes):
    kmeans = KMeans(n_clusters=len(classes), random_state=0).fit(x)
    y_pred = kmeans.predict(x)
    classes = np.asarray(classes)
    y = np.asarray(y,dtype= int)
    x_tmp = pca(x, y, 2)

    f1 = plt.figure(1)
    plt.title("real")
    tmp = plt.scatter(x_tmp[:, 0], x_tmp[:, 1], c=y )
    plt.legend()
    f2 = plt.figure(2)
    plt.title("predicted")
    plt.scatter(x_tmp[:, 0], x_tmp[:, 1], c=y_pred)

    plot_confusion_matrix(y, y_pred, classes, normalize=True)
    plt.show()


def vgg16(model, path: Path, classes, size):
    vgg16_feature_list_np, y_real = load(path, classes, size)

    vgg16_feature_list_np = model.predict(vgg16_feature_list_np)

    clusterize(vgg16_feature_list_np, y_real, classes)


if __name__ == "__main__":
    #  model = ResNet50(weights='imagenet')

    model = MobileNet(weights='imagenet')

    # model = VGG16(weights='imagenet')
    classes = load_classes(base_path.joinpath("train"), count=5)

    vgg16(model, base_path.joinpath("test"), classes, (224, 224))

    x, y = load_my_model(base_path.joinpath("test"), classes)  # , (224, 224), flat=False)
    print(x.shape)
    my_model, features = make_model(x.shape[1:], classes, features=True)

    my_model.fit(x, y, epochs=10)

    y_pred = my_model.predict(x)  # .astype(int)
    print(y_pred)
    # print(np.nonzero(y_pred))
    y_pred = np.array([np.argmax(x) for x in y_pred])

    print(y, y_pred)
    plot_confusion_matrix(y, y_pred, classes)
    plt.show()

    feat = features.predict(x)
    print(feat)

    clusterize(feat, y, classes)
    plt.show()
