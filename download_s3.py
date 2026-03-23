#!/usr/bin/env python3
"""Download all data from the S3 bucket to ./data/"""

import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
)

BUCKET = os.environ["S3_BUCKET"]
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        dest = DATA_DIR / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Skip if local file exists and is same size
        if dest.exists() and dest.stat().st_size == obj["Size"]:
            print(f"  skip {key} (already downloaded)")
            continue

        print(f"  downloading {key} ({obj['Size']:,} bytes)")
        s3.download_file(BUCKET, key, str(dest))

print("Done.")
