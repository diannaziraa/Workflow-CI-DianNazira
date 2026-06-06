FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install mlflow pandas scikit-learn matplotlib

CMD ["python", "MLProject/modelling.py"]