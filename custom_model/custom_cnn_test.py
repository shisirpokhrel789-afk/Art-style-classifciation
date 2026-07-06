import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay

train_ds=tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=(224,224),
    batch_size=32


)

model=(load_model("art.keras"))

valid_ds=tf.keras.utils.image_dataset_from_directory(
    "valid",
    image_size=(224,224),
    batch_size=32
)



y_true=[]
y_pred=[]

class_labels = list(train_ds.class_names)

val_images, val_labels = next(iter(valid_ds))

plt.figure(figsize=(10,10))

for i in range(4):
    plt.subplot(2,2,i+1)

    idx = random.randint(0, len(val_images)-1)

    image = val_images[idx]
    true_label = val_labels[idx].numpy()

    prediction = model.predict(image[np.newaxis,...], verbose=0)
    predicted_label = np.argmax(prediction)

    plt.imshow(image.numpy().astype("uint8"))
    plt.axis("off")

    color = "green" if true_label == predicted_label else "red"

    plt.title(
        f"True: {class_labels[true_label]}\n"
        f"Pred: {class_labels[predicted_label]}",
        color=color
    )

plt.tight_layout()
plt.show()

for images,lables in valid_ds:
    predictions=model.predict(images,verbose=0)
    predicted_labels=np.argmax(predictions,axis=1)
    y_true.extend(lables.numpy()) 
    y_pred.extend(predicted_labels)



cm=confusion_matrix(y_true,y_pred)

class_namnes=valid_ds.class_names
disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_namnes
)

disp.plot(cmap='Blues',xticks_rotation=45)
plt.title("confusion Matrix")
plt.show()


print(classification_report(
    y_true,y_pred,target_names=valid_ds.class_names
))




