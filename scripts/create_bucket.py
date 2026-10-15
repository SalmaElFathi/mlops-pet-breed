import os
import time

import boto3
from botocore.exceptions import ClientError

endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
bucket = os.environ.get("BUCKET_NAME", "mlflow")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="us-east-1",
)

for _attempt in range(30):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"bucket '{bucket}' already exists")
        break
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchBucket", "NotFound"):
            s3.create_bucket(Bucket=bucket)
            print(f"bucket '{bucket}' created")
            break
        raise
    except Exception as error:
        print(f"MinIO not ready ({type(error).__name__}), retrying...")
        time.sleep(2)
else:
    raise SystemExit("MinIO unreachable after 30 attempts")
