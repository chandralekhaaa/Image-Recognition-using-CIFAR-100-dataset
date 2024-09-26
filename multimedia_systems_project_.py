#Importing the required libraries
import numpy as np
import random

#Importing data
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

#Data Statistics
print("Shape of x_train", x_train.shape)
print("Shape of y_train", y_train.shape)
print("Shape of x_test", x_test.shape)
print("Shape of y_test", y_test.shape)

#Image super classes
super_classes = {0:"aquatic_mammals",
                 1:"fish",
                 2:"flowers",
                 3:"food_containers",
                 4:"fruit_and_vegetables",
                 5:"household_electrical_devices",
                 6:"household_furniture",
                 7: "insects",
                 8: "large_carnivores",
                 9: "large_man-made_outdoor_things",
                 10: "large_natural_outdoor_scenes",
                 11: "large_omnivores_and_herbivores",
                 12: "medium_mammals",
                 13: "non-insect_invertebrates",
                 14: "people",
                 15: "reptiles",
                 16: "small_mammals",
                 17: "trees",
                 18: "vehicles_1",
                 19: "vehicles_2"}

#Sample images
import matplotlib.pyplot as plt

for i in range(10):
  random_number = random.randrange(0, 50000)
  img = (x_train[random_number])
  imgplot = plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.title(super_classes[y_train[random_number][0]])
  plt.show()

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses

#Preprocessing y
#One hot encoding
y_train_encoded = tf.one_hot(y_train, 20, dtype=tf.int16)
y_train_encoded = tf.squeeze(y_train_encoded)
y_test_encoded = tf.one_hot(y_test, 20, dtype=tf.int16)
y_test_encoded = tf.squeeze(y_test_encoded)

#Modeling
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(20, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train_encoded, epochs = 20, batch_size=64,
                 verbose=1)

#Model Testing
test_loss, test_acc = model.evaluate(x_test,  y_test_encoded, verbose=1)
print(test_acc)

#Hyperparameter tuning
def buildModel(hyperParameters):
  model = models.Sequential()
  model.add(layers.Conv2D(
      filters=hyperParameters.Int('conv1_filters', min_value=32, max_value=128, step=32),
      kernel_size=hyperParameters.Choice('conv1_kernel', values=[3, 5]),
      activation='relu',
      input_shape=(32, 32, 3)
  ))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(
      filters=hyperParameters.Int('conv2_filters', min_value=64, max_value=256, step=64),
      kernel_size=hyperParameters.Choice('conv2_kernel', values=[3, 5]),
      activation='relu'
  ))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(
      filters=hyperParameters.Int('conv3_filters', min_value=64, max_value=256, step=64),
      kernel_size=hyperParameters.Choice('conv3_kernel', values=[3, 5]),
      activation='relu'
  ))
  model.add(layers.Flatten())
  model.add(layers.Dense(
      units=hyperParameters.Int('dense_units', min_value=32, max_value=128, step=32),
      activation='relu'
  ))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(20, activation='softmax'))
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

!pip install keras-tuner

import kerastuner as kt

tuner = kt.Hyperband(
    buildModel,
    objective='accuracy',
    max_epochs=10,
    factor=3,
    directory='tuning_logs',
    project_name='tuning_logs'
)

import keras

class Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] > 0.95:
            self.model.stop_training = True

tuner.search(
    x=x_train,
    y=y_train_encoded,
    epochs=10,
    callbacks=[Callback()]
)

bestHyperParameters=tuner.get_best_hyperparameters(num_trials=1)[0]

finalModel = tuner.hypermodel.build(bestHyperParameters)
finalModel.fit(x_train, y_train_encoded, epochs=10, validation_split=0.1)

#Modeling using optimal parameters
model = models.Sequential()
model.add(layers.Conv2D(
    filters=64,
    kernel_size= 3,
    activation='relu',
    input_shape=(32, 32, 3)
))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(
    filters=256,
    kernel_size=3,
    activation='relu'
))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(
    filters=256,
    kernel_size=3,
    activation='relu'
))
model.add(layers.Flatten())
model.add(layers.Dense(
    units=128,
    activation='relu'
))
model.add(layers.BatchNormalization())
model.add(layers.Dense(20, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train_encoded, epochs=20, validation_split=0.1)

#Testing and evaluation of the tuned model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis = 1)

accuracy = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)
classificationReport = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusionMatrix)
print("\nClassification Report:\n", classificationReport)

#Pretrained model - Modelling, training, testing
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

baseModel = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

for layer in baseModel.layers:
    layer.trainable = False

model = models.Sequential([
    baseModel,
    layers.GlobalAveragePooling2D(),
    layers.Dense(20, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train_encoded, epochs=30, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test,  y_test_encoded, verbose=1)
print(test_acc)