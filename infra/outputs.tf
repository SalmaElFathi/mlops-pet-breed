output "mlflow_url" {
  description = "MLflow tracking server (use as MLFLOW_TRACKING_URI)."
  value       = "http://localhost:${var.mlflow_port}"
}

output "minio_endpoint" {
  description = "S3 endpoint for DVC and boto3."
  value       = "http://localhost:${var.minio_api_port}"
}

output "minio_console_url" {
  value = "http://localhost:${var.minio_console_port}"
}

output "buckets" {
  value = var.buckets
}
