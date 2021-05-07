from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tensorflow import keras
from keras import layers, models, preprocessing
from PIL import Image
import numpy as np

def read_data(name):
    file_name = "data/" + name + ".txt"
    folder_name = "data/" + name + "/"
    file = open(file_name, "r")
    lines = file.readlines()
    labels = []
    images_png = []
    for line in lines:
        id_image = line.split(',')[0]
        image = folder_name + id_image
        if name == "test":
            image = image[:-1]
        images_png.append(image)

        if name == "test":
            id_image = id_image[:-1]
            labels.append(id_image)
        else:
            labels.append(int(line.split(',')[1]))

    file.close()

    if name != "test":
        labels = np.array(labels).astype(np.int64)

    images = np.array([np.array(Image.open(fname)) for fname in images_png])
    images = images.reshape(images.shape[0], images.shape[1], images.shape[1], 1)
    images = images.astype('float32')

    return labels, images

train_labels, train_images = read_data("train")
validation_labels, validation_images = read_data("validation")
id_images, test_images = read_data("test")

# Importanta normalizarii: datele sa fie aduse la aceeasi scara
# asa incat sa fie compatibile
def normalize_data(train_data, test_data, norm_type):
    if norm_type is None:
        return train_data, test_data

    # Transforma vectorii de caracteristici astfel incat fiecare
    # sa aiba medie 0 si deviatie standard 1
    if norm_type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        return scaler.transform(train_data), scaler.transform(test_data)

    # Transforma fiecare caracteristica individual intre 0 si
    # 1, ceea ce e o metoda buna pentru a pastra valorile de
    # 0 intr-un set de date imprastiat (si acestea pot fi
    # eliminate ulterior)
    if norm_type == 'min_max':
        return (preprocessing.minmax_scale(train_data, axis=-1),
                preprocessing.minmax_scale(test_data, axis=-1))

    # Varianta mai robusta decat L2 deoarece aici se iau doar
    # valorile absolute, deci le trateaza liniar
    if norm_type == 'l1':
        return (preprocessing.normalize(train_data, norm='l1'),
                preprocessing.normalize(test_data, norm='l1'))

    # Varianta mai stabila decat L1 (rezistenta mai mare la
    # ajustari orizontale)
    if norm_type == 'l2':
        return (preprocessing.normalize(train_data, norm='l2'),
                preprocessing.normalize(test_data, norm='l2'))


def test(norm, kernel, c):
    X_train, X_test = normalize_data(train_images, test_images, norm)
    X_train1, X_validation = normalize_data(train_images, validation_images, norm)
    if kernel == 'linear':
        clf = SVC(C = c, kernel = 'linear')
    else:
        clf = SVC(kernel='rbf', gamma=c/10)
    hist = clf.fit(X_train, train_labels)
    preds = clf.predict(X_validation)
    accuracy = accuracy_score(validation_labels, preds)

    print("Accuracy " + norm + " norm with " + kernel + " kernel and " + str(c) +" parameter:", accuracy)


for i in range(3,16,2):
    test('l1', 'rbf', i)
    test('l2', 'rbf', i)
    test('min_max', 'rbf', i)
    test('standard', 'rbf', i)

    test('l1', 'linear', i)
    test('l2', 'linear', i)
    test('min_max', 'linear', i)
    test('standard', 'linear', i)
