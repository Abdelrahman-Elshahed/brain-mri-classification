# Brain MRI Classification (VGG16 & ResNet18)

This project provides a web-based system for classifying brain MRI scans into four categories using deep learning models (VGG16 and ResNet18). It features a FastAPI backend for inference and a Streamlit frontend for user interaction and model comparison.

---

## Table of Contents

  - [Project Structure](#project-structure)
  - [Features](#features)
  - [Usage](#usage)
  - [Setup](#setup)
  - [Model Performance](#model-performance)
  - [Run with Streamlit Application](#run-with-streamlit-application)
  - [PostMan API Testing](#postman-api-testing)
  - [Dockerization](#dockerization)

---

## Project Structure
```bash
brain-mri-classification/
│
├── app/                         # All application code (Streamlit & FastAPI)
│   ├── appfast.py               # FastAPI backend
│   └── new_streamlit_app.py     # Streamlit frontend
│
├── models/                      # Trained model files
│   ├── vgg16_brain_mri_model.h5
│   └── resnet18_weights.pth
│
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build file
│
├── sample_mri_images/           # For MRI images downloaded from the internet to test the model
│
├── notebooks/                   # Jupyter notebooks
    └── Brain_MRI.ipynb
```
---

## Features

- **FastAPI backend** for RESTful model inference (`appfast.py`)
- **Streamlit frontend** for uploading images, viewing predictions, and comparing models (`new_streamlit_app.py`)
- **Pre-trained models**: VGG16 (TensorFlow/Keras) and ResNet18 (PyTorch)
- **Dockerized** for easy deployment

---

## Usage

  - Upload a brain MRI image in the Streamlit app to get predictions from both models.
  - Compare model performance and view detailed metrics and explanations.

---
## Model Performance
   - VGG16: 97% accuracy (macro avg precision: 98%, recall: 97%)
   - ResNet18: 98% accuracy (macro avg precision: 98%, recall: 98%)
---
## Setup

- Clone the Repository

   ```bash
   git clone https://github.com/Abdelrahman-Elshahed/brain-mri-classification.git
   ```
- Create and activate a virtual environment:
  ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
  ```
- Set up dependencies

  Install Python dependencies:
  To install all the required Python packages for the project, run the following command:
  ```bash
  pip install -r requirements.txt
  ```
---
## Run with Streamlit Application

   - Run FastAPI In one terminal
     ```bash
     cd app
     uvicorn appfast:app --reload --host 0.0.0.0 --port 8000
     ```
  - Run Streamlit In another terminal
       ```bash
    cd app
    streamlit run new_streamlit_app.py
     ```
![Image](https://github.com/user-attachments/assets/c22cf0ee-70bc-42bf-b430-4eba0715a9ab)

![Image](https://github.com/user-attachments/assets/a0c37548-1aa8-434d-b4b7-6db37ef0795c)

![Image](https://github.com/user-attachments/assets/6112ec6e-65f2-45e0-bf6a-b48657f6607c)

![Image](https://github.com/user-attachments/assets/786479ec-2a8b-4378-8ae8-ce2a9c1a6f6d)

![Image](https://github.com/user-attachments/assets/66738a0c-3611-43af-87cb-24ba0112c1e3)

---

## PostMan API Testing

- Health Check:
Send a `GET` request to `http://localhost:8000/health` to verify the API is running.

- Model Prediction:
Send a `POST` request to `/predict/vgg` or `/predict/resnet` with a brain MRI image file using the `form-data` key `file`.

![Image](https://github.com/user-attachments/assets/c5384785-6fc0-4e00-a1fc-800d88ae9e84)

![Image](https://github.com/user-attachments/assets/0f2f1861-e461-488e-a51f-e151fc196684)

![Image](https://github.com/user-attachments/assets/77a15611-7d7f-49bb-81a0-9a90e004acfc)

---

## Dockerization

   - Build the Docker image with:
     ```bash
     docker build -t brain-mri-app .
     ```
   - Run the container with:
     ```bash
     docker run -p 8000:8000 -p 8501:8501 brain-mri-app
     ```
  - FastAPI docs: http://localhost:8000/docs
  - Streamlit app: http://localhost:8501

---
