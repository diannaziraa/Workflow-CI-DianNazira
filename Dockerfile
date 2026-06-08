FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends nginx

WORKDIR /opt/mlflow

RUN pip install mlflow==3.13.0

COPY MLProject/model /opt/ml/model

RUN python -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/ml/model', install_mlflow=False, env_manager='local');"
ENV MLFLOW_DISABLE_ENV_CREATION=True
ENV ENABLE_MLSERVER=False
ENV GUNICORN_CMD_ARGS="--timeout 60"

RUN chmod -R 777 /opt/mlflow

ENTRYPOINT ["python","-c","from mlflow.models import container as C; C._serve('local')"]