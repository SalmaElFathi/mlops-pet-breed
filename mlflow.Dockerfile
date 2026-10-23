FROM python:3.12-slim
ARG MLFLOW_VERSION
RUN pip install --no-cache-dir mlflow==${MLFLOW_VERSION} psycopg2-binary boto3
