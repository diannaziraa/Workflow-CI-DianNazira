# Workflow-CI-DianNazira

Proyek ini dibuat untuk memenuhi submission MLOps Dicoding.

## Struktur Proyek

```text
Workflow-CI-DianNazira
├── .github/workflows
│   └── train.yml
├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLproject
│   ├── telco_churn_preprocessed.csv
│   └── DockerHub_Link.txt
├── README.md
└── .gitignore
```

## Workflow CI

Workflow dijalankan secara otomatis menggunakan GitHub Actions setiap kali terdapat push ke branch `main`.

Tahapan workflow:

1. Checkout repository
2. Install dependency
3. Training model
4. Logging MLflow
5. Upload artifact
6. Build Docker Image
7. Push Docker Image ke Docker Hub

## Docker Hub

https://hub.docker.com/repository/docker/diannazira/telco-churn-model/general
