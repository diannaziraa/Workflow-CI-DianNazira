# Workflow-CI-DianNazira

Proyek ini merupakan implementasi Workflow Continuous Integration (CI) untuk tugas akhir kelas **Membangun Sistem Machine Learning** Dicoding.

## Tujuan

Workflow ini digunakan untuk melakukan retraining model Machine Learning secara otomatis menggunakan GitHub Actions dan MLflow Project setiap kali terjadi perubahan pada repository.

## Struktur Project

```text
Workflow-CI-DianNazira
│
├── .github
│   └── workflows
│       └── train.yml
│
├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLproject
│   └── telco_churn_preprocessed.csv
│
├── Dockerfile
└── README.md
```

## Teknologi yang Digunakan

* Python
* Scikit-Learn
* MLflow
* GitHub Actions
* Docker
* Docker Hub

## Workflow

1. Push ke branch main.
2. GitHub Actions otomatis berjalan.
3. MLflow menjalankan training model.
4. Artefak model disimpan.
5. Docker Image dibuat.
6. Docker Image dikirim ke Docker Hub.

## Dataset

Dataset yang digunakan adalah Telco Customer Churn Dataset.

