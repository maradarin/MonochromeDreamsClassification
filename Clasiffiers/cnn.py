import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

tf.random.set_seed(8977)

viz = [0 for i in range(11)]

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

def activation_layers(model, test_images, test_labels, layer):
    viz = [0 for i in range(11)]
    for i in range(100):
        if(viz[int(test_labels[i])] == 0):
            viz[int(test_labels[i])] = 1
            image_batch = np.expand_dims(test_images[i], axis=0)
            layer_outputs = [layer.output for layer in model.layers[:9]]
            activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
            activations = activation_model.predict(image_batch)
            first_layer_activation = activations[0]
            #print(len(first_layer_activation.shape))
            if(len(first_layer_activation.shape) == 3):
                plt.matshow(first_layer_activation[0, :, :], cmap='viridis')
            else:
                plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
            plt.title(str(test_labels[i]) + " - " + str(layer))
            plt.show()

def build_model():
    # Model potrivit pentru crearea unei stive de straturi ce conține strict
    # un strat de input și unul de output
    model = Sequential()

    # Stratul de input (deoarece este primul)
    # Valorile de la input shape corespund dimensiunilor unei imagini
    # width x height x number of channels = 32 x 32 x 1 (grayscale)
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1),
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    activation_layers(model, validation_images, validation_labels, "Input layer")

    # Strat de pooling, folosit pentru a reduce dimensiunile volumului de output
    model.add(MaxPooling2D(pool_size=(2, 2)))
    activation_layers(model, validation_images, validation_labels, "First MaxPool layer, size = 2")

    # Strat convoluțional, am crescut numărul de filtre așa încât modelul
    # să poată învăța caracteristici mai complexe din setul de date
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    activation_layers(model, validation_images, validation_labels, "Conv2D, kernel size = 5")

    # Strat de pooling, folosit pentru a reduce dimensiunile volumului de output
    model.add(MaxPooling2D(pool_size=(2, 2)))
    activation_layers(model, validation_images, validation_labels, "Second MaxPool layer, size = 2")

    model.add(Flatten())
    activation_layers(model, validation_images, validation_labels, "Flatten layer")

    # Strat complet conectat (fiecare neuron primeste input de la toți neuronii din
    # stratul precedent)
    model.add(Dense(750, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    activation_layers(model, validation_images, validation_labels, "Dense layer with 750 neurons")

    # Se renunță în mod aleator la 50% dintre caracteristicile învățate până acum
    model.add(Dropout(0.5))
    activation_layers(model, validation_images, validation_labels, "Dropout layer: 0.5")

    # Strat complet conectat (fiecare neuron primeste input de la toți neuronii din
    # stratul precedent)
    model.add(Dense(250, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    activation_layers(model, validation_images, validation_labels, "Dense layer with 250 neurons")

    # Strat de output, imaginile vor fi distribuite în 9 clase
    # Funcția de activare permite o distribuție de probabilități pe baza
    # căreia se va prezice eticheta imaginii
    model.add(Dense(9, activation='softmax'))
    activation_layers(model, validation_images, validation_labels, "Output layer")

    # Funcția de pierdere: specifică problemelor de multi-clasificare (mai mult de 2 etichete)
    # Funcția de optimizare simulează algoritmul de scădere după gradientul stochastic
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    activation_layers(model, validation_images, validation_labels, "After compilation")

    return model

def fit_and_evaluate(model):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    # Valoarea default a batch size-ului era de 32, dar am ales 64 din cauza gradientului noisy
    hist = model.fit(train_images, train_one_hot, batch_size=64, epochs=100,
                     validation_data=(validation_images, validation_one_hot), callbacks=[es, mc])

    return hist

def make_predictions(model, validation_images, test_images):
    validation_predictions = model.predict(validation_images)
    validation_preds = validation_predictions.argmax(axis=-1)

    test_predictions = model.predict(test_images)
    test_predictions = test_predictions.argmax(axis=-1)
    test_predictions = [str(val) for val in test_predictions]

    my_submission = pd.DataFrame({'id': id_images, 'label': test_predictions})
    my_submission.to_csv('submission.csv', index=False)

    return validation_preds

def plot_graph(hist, type):
    value = "val_" + type
    plt.plot(hist.history[type])
    plt.plot(hist.history[value])
    plt.title('Model ' + type)
    plt.ylabel(type)
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    return [[np.sum((y_true == row) & (y_pred == column))
             for column in range(num_classes)
            ]for row in range(num_classes)]


train_labels, train_images = read_data("train")
validation_labels, validation_images = read_data("validation")
id_images, test_images = read_data("test")

# Vectori binari de forma [0,0,1,0...,0]
# unde vector[i] = 1 dacă imaginea are eticheta
# numărul i, 0 altfel
train_one_hot = to_categorical(train_labels, 9)
validation_one_hot = to_categorical(validation_labels, 9)

# Normalizarea datelor, vrem ca fiecare pixel să se
# afle în intervalul [0, 1)
train_images = train_images / 255
validation_images = validation_images / 255
test_images = test_images / 255

model = build_model()
hist  = fit_and_evaluate(model)
model = load_model('best_model.h5')
model.evaluate(validation_images, validation_one_hot)[1]

validation_preds = make_predictions(model, validation_images, test_images)
cf_matrix = confusion_matrix(validation_labels, validation_preds)
print(classification_report(validation_labels, validation_preds))

plot_graph(hist, 'accuracy')
plot_graph(hist, 'loss')

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
plt.show()
