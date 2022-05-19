from flask import Flask, jsonify, request, Response
from typing import Dict, Union
from io import BytesIO
from PIL import Image
import urllib.request
from gevent import pywsgi
import os
import onnx
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

ort_session: ort.InferenceSession = None

PATH_TO_ONNX_MODEL = os.environ["PATH_TO_ONNX_MODEL"]

def decode_picture_from_url(url: str) -> Union[None, np.array]:
    """Decode a picture from OSS service
    e.g. http://192.168.1.82:9000/mnist/082d97b2-19f1-11ec-a558-1e00d10c4441.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210921%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210921T165246Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=00acbe0487c7dab233d75ec64b3385fccb26dd7c6c8a83858490bdc1e002280e

    Args:
        obj (Dict[str, str]): 
        {
            "bucket_name": "mnist",
            "object_name":"082d97b2-19f1-11ec-a558-1e00d10c4441.png"
        }

        credential (Dict[str, str]): 
        {
            "endpoint": "192.168.1.82:9000",
            "access_key": "testAccessKey",
            "secret_key": "testSecretKey"
        }

    Returns:
        torch.Tensor: Tensor of shape (1,1,28,28)
    """

    try:   
        img = Image.open(BytesIO(urllib.request.urlopen(url, timeout=10).read()))
        img = img.resize((28, 28), Image.ANTIALIAS).convert('L')
        img_np = np.array(img)
        return np.expand_dims(np.expand_dims(img_np, axis=0), axis=0).astype(np.float32) / 255
    except Exception as err:
        print(err)
        return None


@app.route("/run", methods=['POST', 'GET'])
def infer():
    """infer an hand-written digit
    1. Receive json formatted POST request: 
        {
            "bucket_name":"mnist,
            "object_name":"082d97b2-19f1-11ec-a558-1e00d10c4441.png"
        }
    2. Get the Image from OSS
    3. Infer Image
    4. Return the result

    Returns:
        [type]: [description]
    """
    global ort_session
    if ort_session is None: # Neural Network not inited
        init()
        if ort_session is None:
            print("[ Error ] Failed to init neural network:")
            return jsonify({"code": 500, "res": -3})


    try:
        obj_info = request.get_json()["value"] # extract real arguments
        print(obj_info)
    except KeyError:
        print("[ Error ] No argument")
        return jsonify({"code": 500, "res": -2})

    stimulis = decode_picture_from_url(obj_info["url"])
    if stimulis is not None:
        
        stimulis = {ort_session.get_inputs()[0].name:stimulis}
        pred = ort_session.run(None, stimulis)[0]
        res = np.argmax(pred[0])

        print("[ Info ] Prediction tensor is:", pred)
        print("[ Info ] Prediction decoded is:", res)
        return jsonify({"code": 200, "res": int(res)})
    else:
        print("[ Error ] stimulis is None:")
        return jsonify({"code": 500, "res": -1})

@app.route("/init", methods=['POST', 'GET']) # app.route does not accept POST actions by default
def init():
    """Prepare the neural network
    """
    global ort_session

    if ort_session is None:
        model = onnx.load(PATH_TO_ONNX_MODEL)
        onnx.checker.check_model(model)
        ort_session = ort.InferenceSession(PATH_TO_ONNX_MODEL)

    return Response("OK", status=200, mimetype='text/html')


if __name__ == '__main__':
    SERVING_PORT: int = 8080
    init()
    # app.run()
    server = pywsgi.WSGIServer(('0.0.0.0', SERVING_PORT), app)
    server.serve_forever()
    # curl -X POST -d '{"value":{"url":"http://192.168.50.169:9000/mnist/test_picture.png"}}' -H 'Content-Type: application/json' http://localhost:8080/run
    # curl -X POST http://localhost:8080/init