# 🎨 Art Style Classification System

A Deep Learning web application that classifies artwork into one of four artistic styles using Convolutional Neural Networks (CNN) and Transfer Learning (ResNet50). The application is built with **TensorFlow**, **FastAPI**, and **HTML/CSS**, allowing users to upload an artwork and instantly receive the predicted art style along with confidence scores.

---

## 📌 Features

- Upload artwork images through a web interface
- Predict one of four art styles:
  - Minimalism
  - Pop Art
  - Romanticism
  - Symbolism
- Display prediction confidence for every class
- Educational pages describing each art style with example artworks
- FastAPI backend for real-time inference
- Responsive HTML/CSS frontend

---

## 🧠 Models Used

This project contains two different image classification approaches.

### 1. Custom CNN
A Convolutional Neural Network built completely from scratch using TensorFlow/Keras.

### 2. Transfer Learning (ResNet50)
A pretrained ResNet50 model using ImageNet weights for improved feature extraction and classification accuracy.

The deployed web application uses the **ResNet50 model** due to its superior performance.

---

## 📊 Dataset

The model was trained on artwork images belonging to four classes:

- Minimalism
- Pop Art
- Romanticism
- Symbolism

Dataset Split

- Training Set
- Validation Set
- Testing Set

---

## 📈 Model Performance

Transfer Learning (ResNet50)

| Metric | Score |
|---------|-------|
| Accuracy | **80%** |
| Precision | **0.80** |
| Recall | **0.80** |
| F1 Score | **0.80** |

Performance was evaluated using:

- Classification Report
- Confusion Matrix
- Accuracy & Loss Curves

---

## 🛠 Technologies Used

### Machine Learning

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Pillow
- Scikit-learn

### Backend

- FastAPI
- Uvicorn
- Jinja2

### Frontend

- HTML5
- CSS3

---

## 📂 Project Structure

```
Art-Style-Classification/
│
├── custom_model/
│   ├── model.py
│   ├── test.py
│   └── custom_model.keras
│
├── pretrained_model/
│   ├── model.py
│   ├── test.py
│   └── best_resnet.keras
│
├── static/
│   ├── style.css
│   ├── uploads/
│   └── images/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── minimalism.html
│   ├── popart.html
│   ├── romanticism.html
│   └── symbolism.html
│
├── webapp.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Art-Style-Classification.git
```

Move into the project directory

```bash
cd Art-Style-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
uvicorn app:app --reload
```

Open your browser

```
http://127.0.0.1:8000
```

---

## 🎯 How It Works

1. User uploads an artwork image.
2. The image is preprocessed and resized.
3. The trained ResNet50 model extracts image features.
4. The model predicts the artwork style.
5. The website displays:
   - Predicted class
   - Confidence score for every class
   - Uploaded image

---


## 📚 Future Improvements

- Support additional art styles
- Fine-tune the ResNet50 model
- Deploy on AWS EC2
- Docker containerization
- User authentication
- Prediction history
- Mobile responsive UI
- solving overfitting 

---

## 👨‍💻 Author

**Shisir Pokhrel**

Aspiring Machine Learning Engineer | Data Scientist | Python Developer

GitHub:
https://github.com/shisirpokhrel789-afk


---

## ⭐ If you found this project useful, consider giving it a star.

## Dataset Link:
- https://www.kaggle.com/datasets/steubk/wikiart
