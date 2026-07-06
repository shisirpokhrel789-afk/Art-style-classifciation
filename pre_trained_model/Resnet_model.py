#using resent50 model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt
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

base_model=ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)

)

base_model.trainable=False
model=tf.keras.Sequential([
    tf.keras.layers.Lambda(preprocess_input),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation="softmax")])
model.compile(optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

epochs=10
early_stop=EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint=ModelCheckpoint(
    "best_resnet.keras",
    monitor="val_accuracy",
    save_best_only=True
)
history=model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20,
    callbacks=[early_stop,checkpoint])




acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']




#plotting 
epochs_range=range(len(acc))
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)

plt.plot(epochs_range,acc,label='Traninig_accuracy')
plt.plot(epochs_range,val_acc,label='validation_Accuracy')
plt.title('Traninig and validation Accuracy')
plt.show()



