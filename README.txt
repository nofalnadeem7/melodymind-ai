# 🎵 MelodyMind AI
### **End-to-End MLOps Audio Classification System**

MelodyMind is a cloud-native MLOps platform designed to classify music genres (Jazz, Rock, Pop, etc.) using a **Convolutional Recurrent Neural Network (CRNN)**. This project demonstrates a production-ready lifecycle, bridging the gap between deep learning research and scalable deployment using modern DevOps practices.

---

## 🏗️ Architecture

The system follows a microservices pattern, decoupled and orchestrated via **Docker Compose**:

| Service | Stack | Description |
| :--- | :--- | :--- |
| **Frontend** | React (Vite) + Nginx | Modern UI for audio file uploads and real-time inference results. |
| **Backend** | FastAPI + TensorFlow | High-performance API handling audio preprocessing (Librosa) and model serving. |
| **MLflow Server** | Python + SQLite | Centralized Model Registry for versioning, tracking, and lifecycle management. |

---

## 🛠️ Tech Stack

* **Deep Learning**: TensorFlow, Keras (CRNN Architecture)
* **Audio Processing**: Librosa (Mel-spectrogram conversion)
* **API Framework**: FastAPI (Asynchronous Python)
* **Frontend**: React.js, Vite, Tailwind CSS
* **MLOps**: MLflow (Experiment Tracking & Model Registry)
* **Infrastructure**: Docker, Docker Compose, Nginx

---

## 🚀 Getting Started

### 1. Prerequisites
* Ensure you have **Docker Desktop** installed and running.
* No local Python or Node.js runtime is required.

### 2. Infrastructure Setup
Clone the repository and spin up the containers:

```bash
docker compose up -d --build
```

> **Note:** Wait approximately 30–60 seconds for the MLflow database to initialize.

### 3. Model Registration (MLOps Pipeline)
The backend is configured to pull models exclusively from the **Production** stage in MLflow. First, register your trained model:

```bash
docker compose run --rm `
  -v "${PWD}:/mount" `
  -e MLFLOW_URI=http://mlflow:5000 `
  -e MODEL_PATH=/app/models/crnn_net_gru_adam_ours_epoch_40.h5 `
  backend python /mount/register-model.py
```

### 4. Promotion to Production

1. Open the MLflow Dashboard: `http://localhost:5000`
2. Navigate to **Models > MelodyMind_CRNN**.
3. Select the latest version and change its **Stage** to `Production`.
4. Restart the backend to fetch the new production artifact:

```bash
docker compose restart backend
```

---

## 🖥️ Usage

### Web Interface
* Access the application at `http://localhost:3000`.
* Simply drag and drop an audio file (`.wav` or `.mp3`) to see the AI's genre prediction.

### API Documentation
* Explore the interactive Swagger documentation at `http://localhost:8000/docs`.
* You can test the `/predict` endpoint directly through the UI.

### Model Registry
* Monitor and manage your model's lifecycle at `http://localhost:5000`.

---

## 🧠 Model Overview: CRNN

MelodyMind uses a Convolutional Recurrent Neural Network to analyze audio:

* **CNN**: Extracts spatial features and timbral textures from Mel-spectrograms.
* **RNN (GRU)**: Captures temporal dependencies and rhythmic evolution within the track.
* **Preprocessing**: Audio signals are converted to log-scaled Mel-spectrograms, providing a 2D representation of frequency over time.

---

## 📂 Project Structure
├── frontend/           # React frontend source & Nginx configurations
├── backend/            # FastAPI application & ML inference logic
├── models/             # Local directory for model weights (.h5)
├── register-model.py   # Utility script for MLflow registration
└── docker-compose.yml  # Multi-container orchestration manifest

---

*Developed as a demonstration of production-grade MLOps for audio analysis.*
