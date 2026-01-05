# ML Model Inference API (FastAPI)
This repository contains a **FastAPI-based inference service** for a trained machine learning model.  
It demonstrates how an offline-trained ML model can be exposed as a **production-style backend API** for real-time predictions.

---

##  Overview
Machine learning models are rarely useful in isolation. This project focuses on the **serving and integration** side of ML by taking a trained model and deploying it as an API that can be consumed by applications. The model is trained offline, serialized using `joblib`, and loaded into a FastAPI service for inference.

---

## What This Project Demonstrates

- Separation of **training** and **inference** concerns
- Loading a serialized ML model in a backend service
- Exposing predictions via a REST API
- Input validation using Pydantic
- Real-world deployment constraints (environment, inputs, dependencies)

---

##  Project Structure
.
├── app.py
├── breast_cancer_model.joblib
└── README.md

## Running the Service Locally
### Start the API
uvicorn app:app --reload

#### The service will be available at:
http://127.0.0.1:8000/docs

# Kaggle Case Study

The full case study explaining the training process, deployment decisions, and key learnings is documented in a Kaggle notebook:
🔗 Kaggle Notebook:
https://www.kaggle.com/code/angelchaudhary/case-study-from-model-to-api

This repository serves as the deployment companion to that notebook.


