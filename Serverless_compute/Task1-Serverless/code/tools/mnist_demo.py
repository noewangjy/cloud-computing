import argparse
from typing import Union, Dict
from file_uploader import oss_upload
import requests
import json
from minio import Minio
import uuid
import os
from datetime import timedelta


def invoke_openwsk_action_mnist(wsk_apihost: str, image_url: str):
    """Invoke openwsk action via RESTful API

    Args:
        wsk_apihost (str): url like string 
        image_url (str): url to image
    """
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=wsk_apihost,
                             headers=headers,
                             data=json.dumps({"url": image_url}))
    return response.json()


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
    parser.add_argument('--mnist_api', type=str)
    args = parser.parse_args()

    minioClient = Minio(args.endpoint,
                        access_key=args.access_key,
                        secret_key=args.secret_key,
                        secure=False)  # Create MinIO client

    if len(args.object_name) <= 0:
        # Generate unique object name
        object_name = str(uuid.uuid1()) + os.path.splitext(args.file)[-1]
    else:
        object_name = args.object_name

    # Upload image
    minioClient.fput_object(args.bucket_name, 
                            object_name,
                            args.file)
    img_url = minioClient.presigned_get_object(args.bucket_name,
                                           object_name,
                                           expires=timedelta(days=2))

    # Invoke OpenWhisk via api
    res = invoke_openwsk_action_mnist(args.mnist_api, img_url)
    print(res)

    # Delete the image
    minioClient.remove_object(args.bucket_name, object_name)
