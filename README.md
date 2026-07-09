# Garbage_Classification
# ♻️ Garbage Classification using CNN

## 📌 Project Overview

This project is a Deep Learning-based Garbage Classification System developed using Convolutional Neural Networks (CNN). The model classifies uploaded garbage images into different waste categories, helping automate waste sorting and promote proper recycling.

A Streamlit dashboard is included for an interactive user experience.

---

## 🚀 Features

- Image Classification using CNN
- Upload Garbage Images
- Predict Garbage Category
- Display Prediction Confidence
- Interactive Streamlit Dashboard
- Easy-to-use Interface

---

## 🗂 Dataset

The dataset contains the following garbage categories:

- Battery
- Cardboard
- Glass
- Paper
- Plastic

> Images are organized into separate folders for each class.

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Streamlit
- Pillow

---

## 📂 Project Structure

```
Garbage_Classification/
│
├── dataset/
│   ├── Battery/
│   ├── Cardboard/
│   ├── Glass/
│   ├── Paper/
│   └── Plastic/
│
├── model/
│   └── garbage_classifier.keras
│
├── test_image/
│   └── plastic.jpg
│
├── train_model.py
├── test_model.py
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

- Conv2D
- BatchNormalization
- MaxPooling2D
- Dropout
- Flatten
- Dense
- Softmax Output Layer

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone <repository-link>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

### 4. Test the Model

```bash
python test_model.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 Output

The application predicts one of the following classes:

- Battery
- Cardboard
- Glass
- Paper
- Plastic

It also displays the prediction confidence.

---

## 📈 Future Improvements

- Increase dataset size
- Improve CNN architecture
- Add more garbage categories
- Real-time webcam classification
- Deploy using Streamlit Cloud

---

## 👨‍💻 Author

**Irnitee C. Patel**

B.Tech Computer Engineering (Artificial Intelligence)
