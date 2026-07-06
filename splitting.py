
import os

import shutil
from pathlib import Path
import random

root_folders=["test","train","valid"]
folders=["Minimalism","Pop_Art","Romanticism","symbolism"]

for root in root_folders:
    for folder in folders:
        os.makedirs(os.path.join(root,folder))



print(len("test/Minimalism"))

base = Path(r"C:\Users\Dell\.cache\kagglehub\datasets\steubk\wikiart\versions\1")

SOURCE_PATH = [
    base / "Minimalism",
    base / "Pop_Art",
    base / "Romanticism",
    base / "Symbolism"
]

DESTINATION_PATH=["Minimalism","Pop_Art","Romanticism","symbolism"]

for src,dst in zip(SOURCE_PATH,DESTINATION_PATH):
    files=os.listdir(src)

    #shuffle so you don't always get the same images
    random.shuffle(files)
    train_files=files[:250]
    valid_files=files[250:300]
    test_files=files[300:350]
    
    #copying training images
    for file in train_files:
        shutil.copy(
            os.path.join(src,file),
            os.path.join("train",dst,file)
        )
    #copying validation images
    for file in valid_files:
        shutil.copy(
            os.path.join(src,file),
            os.path.join("valid",dst,file)
        )
    #copying testing images
    for file in test_files:
        shutil.copy(
            os.path.join(src,file),
            os.path.join("test",dst,file)
        )
print("Done!")
