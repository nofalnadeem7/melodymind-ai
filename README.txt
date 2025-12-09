🎵 MelodyMind: End-to-End MLOps Audio Classifier

MelodyMind is a cloud-native MLOps project that classifies audio files into music genres (Jazz, Rock, Pop, etc.) using a CRNN (Convolutional Recurrent Neural Network).
The project is fully containerized and demonstrates a production-grade pipeline using React, FastAPI, TensorFlow, and MLflow for model lifecycle management.

🏗️ Architecture

The system consists of three microservices orchestrated using Docker Compose:

🔹 Frontend (Port 3000)

Built with React (Vite)

Served with Nginx

Acts as a reverse proxy for secure API calls

🔹 Backend (Port 8000)

FastAPI service running on Python 3.11

Handles audio preprocessing using Librosa

Performs genre inference using TensorFlow models

🔹 MLflow Server (Port 5000)

Central Model Registry

Stores versioned models

Backend loads only models tagged as Production

🚀 Prerequisites

You only need:

Docker Desktop (installed and running)

No local Python or Node.js installation required.

⚡ Quick Start Guide

Follow the steps exactly to deploy the full application.

1. Start the Infrastructure

Open PowerShell in the project folder and run:

docker compose up -d --build


Wait 30–60 seconds for all containers to fully initialize.

2. Register the Model (MLOps Step)

The backend requires a trained model.
Use the script below to upload the model to MLflow and fix layer weights:

docker compose run --rm `
  -v "${PWD}:/mount" `
  -e MLFLOW_URI=http://mlflow:5000 `
  -e MODEL_PATH=/app/models/crnn_net_gru_adam_ours_epoch_40.h5 `
  backend python /mount/register-model.py


If successful, you will see:

Success! Model Registered.

3. Promote the Model to Production

Open the MLflow UI:
http://localhost:5000

Navigate to:
Models → MelodyMind_CRNN

Select Version 1

From the "Stage" dropdown, choose Production

Confirm the transition

4. Restart Backend

Restart the backend so it downloads the production model:

docker compose restart backend

🖥️ How to Use
1. Web App (Frontend)

URL: http://localhost:3000

Action: Upload an audio file (.wav, .mp3) → Click Predict

2. API Documentation (Swagger UI)

URL: http://localhost:8000/docs

Action: Test the /predict endpoint directly

3. MLflow Model Registry

URL: http://localhost:5000

Action: Inspect model versions, stages, metrics, and artifacts
