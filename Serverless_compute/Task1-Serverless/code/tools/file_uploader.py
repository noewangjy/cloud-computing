from minio import Minio
import argparse
import uuid
from typing import Union, Dict
import os
from datetime import timedelta


def oss_upload(endpoint: str,
               bucket_name: str,
               access_key: str,
               secret_key: str,
               file: str,
               object_name: str = '') -> Union[None, Dict[str, str]]:
    """Upload a file to MinIO OSS

    Args:
        args (ArgumentParser): should contain following attributes:
            - args.endpoint: str, url
            - args.access_key: str
            - args.secret_key: str

    Returns:
        Dict[str, str]: {$BUCKET:$NAME}
    """
    minioClient = Minio(endpoint,
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=False)  # Create MinIO client
    try:
        if len(object_name) <= 0:
            object_name = str(uuid.uuid1()) + os.path.splitext(file)[
                -1]  # Generate unique object name
        else:
            object_name = object_name

        minioClient.fput_object(bucket_name, object_name,
                                file)  # Upload object
        url = minioClient.presigned_get_object(bucket_name,
                                               object_name,
                                               expires=timedelta(days=2))
        ret = {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "url": url
        }  # Return the object info
        return ret

    except Exception as err:
        print(err)
        return None


if __name__ == '__main__':
    """Usage

    python file_upload.py --endpoint=192.168.1.82:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket=mnist --file=mnist.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str)
    parser.add_argument('--access_key', type=str)
    parser.add_argument('--secret_key', type=str)
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--file', type=str)
    parser.add_argument('--object_name', type=str, default='')
    args = parser.parse_args()
    print(
        oss_upload(endpoint=args.endpoint,
                   bucket_name=args.bucket_name,
                   access_key=args.access_key,
                   secret_key=args.secret_key,
                   file=args.file,
                   object_name=args.object_name))
