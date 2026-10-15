import mlflow

from prodml.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="premier-run"):
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("top1", 0.5)
    with open("note.txt", "w") as f:
        f.write("hello")
    mlflow.log_artifact("note.txt")
print("run envoyé")
