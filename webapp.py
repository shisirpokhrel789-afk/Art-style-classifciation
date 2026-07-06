from fastapi import FastAPI,Response,Request,File,UploadFile,HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


app=FastAPI(title="Ar -style classification")
app.mount("/static",StaticFiles(directory="static"),name="static")
app.mount("/uploads",StaticFiles(directory="uploads"),name="uploads")
app.mount("/test",StaticFiles(directory="test"),name="test")
templates=Jinja2Templates(directory="templates")
# UPLOAD_DIR=os.makedirs("Art classification model","uploads")

base_model=load_model("pre_trained_model/best_resnet.keras",custom_objects={"preprocess_input":preprocess_input})

train_ds=train_ds=tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=(224,224),
    batch_size=32


)



@app.get("/")
def home(request:Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request":request,
            "name":"Art-style Prediction",
            "context":"This system classfies the art style of the uploaded image of the artCategories are:Minimalism,Pop_Art,Romanticism,Symbolism"

        }
    )


@app.get("/minimalism")
def minimalism(request:Request):
    return templates.TemplateResponse(
        "Minimalism.html",
        {'request':request}
    )
@app.get("/popart")
def popart(request:Request):
    return templates.TemplateResponse(
        "PopArt.html",
        {'request':request}
    )
@app.get("/romanticism")
def romanticism(request:Request):
    return templates.TemplateResponse(
        "Romanticism.html",
        {'request':request}
    )
@app.get("/symbolism")
def symbolism(request:Request):
    return templates.TemplateResponse(
        "Symbolism.html",
        {'request':request}
    )

@app.post("/upload")
def upload(
    request:Request,
    image:UploadFile= File(...)
):
    
    #prevent upload folder to grow
    for file in os.listdir("uploads"):
        file_path = os.path.join("uploads", file)

        if os.path.isfile(file_path):
            os.remove(file_path)
    path=f'uploads/{image.filename}'
    with open(path,"wb") as buffer:
        shutil.copyfileobj(image.file,buffer)
    img=tf.keras.preprocessing.image.load_img(
        path,target_size=(224,224)
    )
    img=tf.keras.preprocessing.image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)

    prediction=base_model.predict(img)

    probabilites=prediction[0]
    predicted_index=np.argmax(prediction)
    
    class_names = list(train_ds.class_names)

    predicted_class=class_names[predicted_index]
    
    
    return templates.TemplateResponse(
        "index.html",{
            "request":request,
            "image_url":f"/uploads/{image.filename}",
            "prediction":predicted_class,
            "probabilities":zip(class_names,probabilites)
        }
    )










    



    



