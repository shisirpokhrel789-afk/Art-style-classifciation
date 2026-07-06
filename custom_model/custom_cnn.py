import os
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

train_ds=tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=(224,224),
    batch_size=32


)
valid_ds=tf.keras.utils.image_dataset_from_directory(
    "valid",
    image_size=(224,224),
    batch_size=32
)

test_ds=tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=(224,224),
    batch_size=32,
    shuffle=False

)

# print(train_ds.class_names)



#adding normalization and agumentation
normlization=tf.keras.layers.Rescaling(1./255)

data_agumentation=Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)

])

model=Sequential([

    tf.keras.Input(shape=(224,224,3)),

    data_agumentation,
    normlization,

    Conv2D(32,(3,3) ,activation="relu"),
   
    MaxPooling2D(),

    Conv2D(64,(3,3),activation="relu"),
    
    MaxPooling2D(),

    Conv2D(128,(3,3),activation="relu"),
   
    MaxPooling2D(),

    Conv2D(256,(3,3),activation="relu"),
   
    MaxPooling2D(),

    Flatten(),

    Dense(128,activation="relu"),
    Dropout(0.5),

    Dense(4,activation="softmax")])


epochs=20
#compliation of model
model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

#Training the mode
history=model.fit(train_ds,validation_data=valid_ds,epochs=epochs)
model.save("art.keras")


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range=range(epochs)


#plotting 
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)

plt.plot(epochs_range,acc,label='Traninig_accuracy')
plt.plot(epochs_range,val_acc,label='validation_Accuracy')
plt.title('Traninig and validation Accuracy')
plt.show()

#testing the 
