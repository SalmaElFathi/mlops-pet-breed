variable "prefix" {
  description = "Prefix for every resource name, so this environment lives next to the Compose one."
  type        = string
  default     = "tf"
}

variable "postgres_image" {
  type    = string
  default = "postgres:16"
}

variable "minio_image" {
  type    = string
  default = "cgr.dev/chainguard/minio@sha256:2136d593beacf4461187035eca949fe49abd589d5079735025c711ca58fee0c8"
}

variable "mlflow_image" {
  description = "Local image built from mlflow.Dockerfile (docker compose build mlflow)."
  type        = string
  default     = "pet-mlflow"
}

variable "postgres_user" {
  type    = string
  default = "mlflow"
}

variable "postgres_db" {
  type    = string
  default = "mlflow"
}

variable "postgres_password" {
  type      = string
  sensitive = true
}

variable "minio_user" {
  type    = string
  default = "minio_admin"
}

variable "minio_password" {
  type      = string
  sensitive = true
}

variable "buckets" {
  description = "Buckets to create: MLflow artifacts and the DVC remote."
  type        = list(string)
  default     = ["mlflow", "dvc"]
}

variable "mlflow_port" {
  type    = number
  default = 15000
}

variable "minio_api_port" {
  type    = number
  default = 19000
}

variable "minio_console_port" {
  type    = number
  default = 19001
}
