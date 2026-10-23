locals {
  bucket_script = abspath("${path.module}/../scripts/create_bucket.py")
}

resource "docker_network" "mlops" {
  name = "${var.prefix}-net"
}

resource "docker_volume" "pgdata" {
  name = "${var.prefix}-pgdata"
}

resource "docker_volume" "miniodata" {
  name = "${var.prefix}-miniodata"
}

resource "docker_image" "postgres" {
  name         = var.postgres_image
  keep_locally = true
}

resource "docker_image" "minio" {
  name         = var.minio_image
  keep_locally = true
}

data "docker_image" "mlflow" {
  name = var.mlflow_image
}

resource "docker_container" "postgres" {
  name    = "${var.prefix}-postgres"
  image   = docker_image.postgres.image_id
  restart = "unless-stopped"
  wait    = true

  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=${var.postgres_db}",
  ]

  healthcheck {
    test     = ["CMD-SHELL", "pg_isready -U ${var.postgres_user} -d ${var.postgres_db}"]
    interval = "5s"
    timeout  = "3s"
    retries  = 10
  }

  networks_advanced {
    name    = docker_network.mlops.name
    aliases = ["postgres"]
  }

  volumes {
    volume_name    = docker_volume.pgdata.name
    container_path = "/var/lib/postgresql/data"
  }
}

resource "docker_container" "minio" {
  name    = "${var.prefix}-minio"
  image   = docker_image.minio.image_id
  restart = "unless-stopped"
  command = ["server", "/data", "--console-address", ":9001"]

  env = [
    "MINIO_ROOT_USER=${var.minio_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_password}",
  ]

  ports {
    internal = 9000
    external = var.minio_api_port
  }

  ports {
    internal = 9001
    external = var.minio_console_port
  }

  networks_advanced {
    name    = docker_network.mlops.name
    aliases = ["minio"]
  }

  volumes {
    volume_name    = docker_volume.miniodata.name
    container_path = "/data"
  }
}

# One short-lived container per bucket: it retries until MinIO answers, then exits.
resource "docker_container" "create_bucket" {
  for_each = toset(var.buckets)

  name     = "${var.prefix}-create-bucket-${each.key}"
  image    = data.docker_image.mlflow.id
  restart  = "no"
  must_run = false
  logs     = true
  command  = ["python", "/create_bucket.py"]

  env = [
    "MLFLOW_S3_ENDPOINT_URL=http://minio:9000",
    "AWS_ACCESS_KEY_ID=${var.minio_user}",
    "AWS_SECRET_ACCESS_KEY=${var.minio_password}",
    "BUCKET_NAME=${each.key}",
  ]

  networks_advanced {
    name = docker_network.mlops.name
  }

  volumes {
    host_path      = local.bucket_script
    container_path = "/create_bucket.py"
    read_only      = true
  }

  depends_on = [docker_container.minio]
}

resource "docker_container" "mlflow" {
  name    = "${var.prefix}-mlflow"
  image   = data.docker_image.mlflow.id
  restart = "unless-stopped"

  command = [
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5000",
    "--backend-store-uri",
    "postgresql+psycopg2://${var.postgres_user}:${var.postgres_password}@postgres:5432/${var.postgres_db}",
    "--artifacts-destination", "s3://mlflow",
    "--serve-artifacts",
  ]

  env = [
    "MLFLOW_S3_ENDPOINT_URL=http://minio:9000",
    "AWS_ACCESS_KEY_ID=${var.minio_user}",
    "AWS_SECRET_ACCESS_KEY=${var.minio_password}",
    "MLFLOW_SERVER_ALLOWED_HOSTS=mlflow:5000,localhost:*,127.0.0.1:*",
  ]

  ports {
    internal = 5000
    external = var.mlflow_port
  }

  networks_advanced {
    name    = docker_network.mlops.name
    aliases = ["mlflow"]
  }

  depends_on = [docker_container.postgres, docker_container.minio]
}
