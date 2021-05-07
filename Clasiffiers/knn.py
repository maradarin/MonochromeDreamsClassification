import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import imageio

def read_data(name):
    file_name = name + ".txt"
    folder_name = name + "/"
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

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_images(self, test_image, num_neighbors, metric):
        # pt fiecare imagine test i, avem cate o distanta
        if metric == 'l2':
            distances = np.sum((train_images - test_image) ** 2, axis=-1)
        elif metric == 'l1':
            distances = np.sum(np.abs(train_images - test_image), axis=-1)

        # sortare crescatoare dupa distante, sunt reordonati si returnati indecsii
        # functia argsort returneaza indecsii care ordoneaza lista
        sorted_indexes = np.argsort(distances)
        # sunt luate primele 3 etichete din indecsii sortati
        top_neighbors = self.train_labels[sorted_indexes[:num_neighbors]]
        # functia bincount calculeaza nr de aparitii al fiecarei valori din lista
        class_counts = np.bincount(top_neighbors)

        return np.argmax(class_counts)


clf = KnnClassifier(train_images, train_labels)

def test(k, metric):
    predictions = []
    for validation_image in validation_images:
        pred_label = clf.classify_images(validation_image, k, metric)
        predictions.append(pred_label)

    pred_labels = np.array(predictions)
    correct_count = np.sum(pred_labels == validation_labels)
    total_count = len(validation_labels)

    accuracy = correct_count / total_count

    print("Accuracy for " + str(k) + " neighbours and " + metric + " metric: " + str(accuracy*100) + "%")


for k in range(3, 28, 2):
    test(k, "l1")
    test(k, "l2")