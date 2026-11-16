# Heart Disease Prediction API
### FastAPI + Machine Learning + Docker + Render

A simple machine learning-powered REST API that predicts the presence or absence of heart disease using Logistic Regression. The application is built with FastAPI, containerized using Docker, and deployed on Render.

This project demonstrates machine learning model training, REST API development, Docker containerization, and cloud deployment.

---

## Project Overview

The Heart Disease Prediction API uses a trained Logistic Regression classifier to predict whether a person is likely to have heart disease based on 13 medical input features.

**Key Features:**
- Machine learning classification using Scikit-learn.
- REST API developed with FastAPI.
- Input validation using Pydantic.
- Model serialization using Joblib.
- Docker containerization.
- Docker Compose support.
- Cloud deployment using Render.
- Interactive API documentation with Swagger UI.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| FastAPI | REST API framework |
| Scikit-learn | Machine learning |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Pydantic | Input validation |
| Joblib | Model serialization |
| Uvicorn | ASGI server |
| Docker | Application containerization |
| Docker Compose | Container management |
| Render | Cloud deployment |

---

## Project Structure

```text
heart-disease-prediction-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── heart.csv
│
├── model/
│   ├── README.txt
│   └── heart_model.joblib
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py
├── .dockerignore
└── README.md
```

---

## 1. Dataset

**Dataset:** Heart Disease Dataset

**Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

The dataset contains patient health information used to train a binary classification model.

The model uses 13 input features:

1. age
2. sex
3. cp
4. trestbps
5. chol
6. fbs
7. restecg
8. thalach
9. exang
10. oldpeak
11. slope
12. ca
13. thal

**Target variable:** `target`

- `0` = Absence of heart disease
- `1` = Presence of heart disease

---

## 2. Machine Learning Model

**Algorithm:** Logistic Regression

The model is trained using Scikit-learn and saved using Joblib.

**Model file:**

```text
model/heart_model.joblib
```

**Model Performance:**

| Metric | Result |
|---|---:|
| Algorithm | Logistic Regression |
| Test Accuracy | 80.3% |
| Test Samples | 61 |
| Dataset | Heart Disease Dataset |

The trained model is loaded by the FastAPI application to generate predictions.

---

## 3. FastAPI Application

The application provides three main API endpoints.

### API Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Health and model-loaded status |
| GET | `/info` | Model description and feature list |
| POST | `/predict` | Predict binary heart disease output |

### Prediction Input Example

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

### Prediction Response Example

```json
{
  "heart_disease": true,
  "prediction": 1,
  "probability": 0.8745
}
```

The prediction and probability depend on the submitted input.

`heart_disease: true` means the model predicts target class 1.

---

## 4. Local Installation and Setup

### Prerequisites

- Python 3.12
- pip
- Git
- Docker Desktop (for Docker execution)

### Step 1: Clone the Repository

```bash
git clone https://github.com/syedhamim-swe/heart-disease-prediction-api.git
```

Navigate to the project directory:

```bash
cd heart-disease-prediction-api
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

Ensure the dataset is available at:

```text
data/heart.csv
```

Run:

```bash
python train.py
```

The trained model will be saved at:

```text
model/heart_model.joblib
```

The generated model file must be committed to the GitHub repository before building the Docker image.

### Step 5: Run FastAPI Locally

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

http://localhost:8000/docs

---

## 5. Docker Configuration

The application is containerized using Docker.

The Dockerfile:
- Uses Python 3.12 slim as the base image.
- Installs dependencies from requirements.txt.
- Copies the FastAPI application and trained model.
- Exposes port 8000 for local use.
- Starts the application using Uvicorn.
- Supports Render's dynamically assigned PORT environment variable.

### Build Docker Image

```bash
docker compose build
```

### Run the Application

```bash
docker compose up
```

### Stop the Application

```bash
docker compose down
```

### View Container Logs

```bash
docker compose logs
```

Access Swagger UI locally:

http://localhost:8000/docs

---

## 6. Deploy to Render

The application is deployed on Render using Docker.

### Deployment Steps

1. Push the complete project to a GitHub repository. Include `model/heart_model.joblib`; it is generated after training.
2. In Render, choose **New + → Web Service** and connect the repository.
3. Set **Runtime/Environment** to **Docker**. Set the root directory to the repository root if the Dockerfile is there.
4. Deploy. Render detects the Dockerfile and builds the image. The application listens on `0.0.0.0` using the `PORT` environment variable provided by Render (defaulting to `8000` for local use).
5. When deployment finishes, test the `/health`, `/info`, and `/docs` endpoints using your Render service URL.

Do not set a separate start command unless you intentionally override the Dockerfile CMD.

---

## 7. Live Deployment

**Platform:** Render

**Live API URL:**  
https://heart-disease-prediction-api-rejh.onrender.com

**Swagger Documentation:**  
https://heart-disease-prediction-api-rejh.onrender.com/docs

**Health Check:**  
https://heart-disease-prediction-api-rejh.onrender.com/health

**Model Information:**  
https://heart-disease-prediction-api-rejh.onrender.com/info

---

## 8. Troubleshooting

- **Model file missing:** Run `python train.py` and ensure `model/heart_model.joblib` exists before building the Docker image.
- **CSV columns differ:** Inspect the downloaded CSV headers and update `FEATURES`/`required` in `train.py` to match. This solution expects the common Kaggle file with a `target` column.
- **Container exits on startup:** Inspect logs with `docker compose logs`.
- **Port already in use:** Change the left side of the Compose mapping, e.g. `8001:8000`, then open `http://localhost:8001/docs`.
- **Render service sleeping:** Free Render instances may spin down after inactivity. The first request can take approximately 50 seconds or more to respond.

---

## 9. Model Performance

The Logistic Regression model achieved approximately **80.3% test accuracy** on the evaluation dataset.

This project focuses primarily on demonstrating:
- Machine learning model development.
- FastAPI application development.
- Docker containerization.
- Docker Compose execution.
- Cloud deployment using Render.

---

## 10. Disclaimer

This project is developed for educational purposes only.

The machine learning model is not a medical device and must not be used to diagnose, treat, or make healthcare decisions.

Predictions are based on a trained classification model and should not be considered medical advice.

---

## Author

**Syed Hamim**

GitHub: https://github.com/syedhamim-swe

**Project:** Heart Disease Prediction API

**Deployment:** Render
