# Heart Disease Prediction – End-to-End MLOps Pipeline

This repository implements an end-to-end MLOps workflow for predicting the risk of heart disease using the UCI Heart Disease dataset.  
The project demonstrates real-world MLOps practices including data preprocessing, model training, CI/CD automation, containerization, Kubernetes deployment, and monitoring.

---

## 🎯Problem Statement

Build a machine learning classifier to predict the presence or absence of heart disease based on patient health data and deploy the solution as a production-ready, monitored API.

---

## 📊 Dataset

- **Name**: Heart Disease UCI Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: 13 clinical attributes (age, sex, cholesterol, blood pressure, etc.)
- **Target**: Binary classification (0 = No heart disease, 1 = Heart disease)

Dataset file:
- `heart_disease_data.csv` : [Link to dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) 


---

## 🧠 Model & Training

- Algorithm: Logistic Regression
- Preprocessing:
  - Handling missing values (`?` → NaN → median imputation)
  - Feature scaling using `StandardScaler`
- Training pipeline implemented using Scikit-learn
- Model serialized using `joblib`

> Exploratory analysis and experimentation were performed in Google Colab.  
> Final model training and serialization were executed in the deployment environment to ensure reproducibility.

---

## ⚙️ Tech Stack

- **Python**: 3.8+ (local), 3.10 (Docker & CI)
- **ML**: Scikit-learn
- **API**: FastAPI
- **Containerization**: Docker
- **Orchestration**: Kubernetes (Minikube)
- **CI/CD**: GitHub Actions
- **Monitoring**: Application logs + health endpoint

---

## 📁 Repository Structure

.
├── app.py
├── train_model.py
├── test_model.py
├── requirements.txt
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── artifacts/
│ └── heart_model.pkl
├── data/
│ └── heart_disease.csv
└── .github/workflows/ci.yml

---
## 🧩 Prerequisites

Ensure the following are installed:

- Git
- Docker Desktop (WSL2 enabled on Windows)
- Minikube
- kubectl
- Python 3.8+ (optional if using Docker)

Verify installations:

```bash
docker --version
minikube version
kubectl version --client 
```

## 📥Clone the repository
```bash
git clone https://github.com/mohankr30/MLOPs_Assignment1_Group72.git
cd MLOPS_Assignment1_Group72
```

## 🐳 Build Docker Image
Ensure Docker Desktop is running.
```bash
docker build -t heart-api:1.0 .
```
Verify image:
```bash
docker images
```
## ☸️ Start Local Kubernetes Cluster
```bash
minikube start --driver=docker
```
Verify:
```bash
kubectl get nodes
```
Expected:
```bash
minikube   Ready    <IP_ADDRESS>   <none>        <none>       Ready
```

## 📦 Load Docker Image into Minikube
```bash 
minikube image load heart-api:1.0
``` 
## 🚀 Deploy to Kubernetes & Apply Service
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```
Verify pods:
```bash
kubectl get pods
```
Expected:
```bash 
heart-api-xxxxx   Running
```
## 🌐 Access the Deployed API(Local)
Get service URL:
```bash
minikube service heart-api-service --url
```
Example output:
```bash
http:////192.168.49.2:31234
```
⚠️ The IP and port will differ per machine.

## 📖 Swagger UI

Open in browser:
```
http://<MINIKUBE_URL>/docs
```
This provides interactive API documentation.

## 🔮 Prediction Endpoint
Endpoint: 

```POST /predict```

Sample Request: 

````
{
  "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}
````

Sample Response:
```
{
  "prediction": 0,
  "confidence": 0.03
}
```

## ❤️ Health Check

```
curl http://<MINIKUBE_URL>/health
```


Response:

```
{"status":"ok"}
```

## 📊 Monitoring & Logging

* Application-level logging implemented using FastAPI middleware
* Logs include:
    * Incoming requests
    * Response status codes
    * Request processing time

View logs:
```bash
kubectl logs deployment/heart-api
```

## 🔒 Access Scope

* The API is deployed on local Kubernetes (Minikube)
* Endpoint is accessible only from the host machine
* Public access would require deployment to a managed cloud Kubernetes service (EKS, GKE, AKS)

## 🔁 CI/CD Pipeline

A GitHub Actions pipeline is implemented to automate:

* Linting using flake8
* Unit testing using pytest
* Model training
* Artifact upload (heart_model.pkl)

Pipeline runs on every push and pull request and provides logs and artifacts per run.

## 📌 Notes for Users

* Local Kubernetes deployment using Minikube satisfies assignment requirements
* All endpoints were verified and logged
* Screenshots of CI/CD, deployment, and predictions are provided
* The project follows reproducible and production-aligned MLOps practices
---
👤 Authors

Mohan K R (2024aa05419@wilp.bits-pilani.ac.in)  
Shreyas T S (2024aa05418@wilp.bits-pilani.ac.in)




