"""run_server.py"""
import argparse
import os
import multiprocessing as mp
from threading import Lock
import pickle
import json
import queue
import random

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import numpy as np


from gevent import pywsgi
from flask import Flask, request, Response
from typing import Dict, Any, Union


from models import Net

RANDOM_SEED=0
torch.manual_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

app = Flask(__name__)

g_net: Union[None, nn.Module] = None
g_lock: Lock = Lock()
g_workers: Dict[Any, str] = None
g_gradient_queue: mp.Queue = None
g_optimizer: Optimizer = None
g_policy: Dict[str, Any] = None
g_version: int = None

@app.route("/registerWorker", methods=['POST'])
def register_worker() -> Response:
    """Get worker registration info, and register worker

    Worker send json formated registration info:
    {
        "id":$WORKER_UNIQUE_ID
        "description":$WORKER_DESCRIPTION
    }
    Returns:
        Response: json response of status
    
    TODO: 
        use access_key and secret_key to protect this interface
    """
    resp = Response(mimetype='application/json')
    try:
        worker_info: Dict[str, Any] = request.get_json()
    except KeyError:
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if "id" not in worker_info.keys():
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if worker_info["id"] in g_workers.keys():
        resp.status = 409
        resp.data = json.dumps({"code": 409, "msg": "The worker has already registered"})
        return resp

    else:
        if "description" in worker_info.keys():
            g_workers[worker_info["id"]] = str(worker_info["description"])
        else:
            g_workers[worker_info["id"]] = ''

        resp.status = 200
        resp.data = json.dumps({"code": 200, "msg": "Successfully registered worker"})
        return resp


@app.route("/unregisterWorker", methods=['POST'])
def unregister_worker() -> Response:
    """Unregister a worker

    Worker send json formated registration info:
    {
        "id":$WORKER_UNIQUE_ID
    }
    Returns:
        Response: json response of status
    
    TODO: 
        use access_key and secret_key to protect this interface
    """
    resp = Response(mimetype='application/json')
    try:
        worker_info: Dict[str, Any] = request.get_json()
    except KeyError:
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if "id" not in worker_info.keys():
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if worker_info["id"] not in g_workers.keys():
        resp.status = 409
        resp.data = json.dumps({"code": 409, "msg": "The worker has not registered"})
        return resp

    else:
        del g_workers[worker_info["id"]]
        resp.status = 200
        resp.data = json.dumps({"code": 200, "msg": "Successfully unregistered worker"})
        return resp


@app.route("/getAccuracy", methods=['GET'])
def get_accuracy():
    # TODO: Implement this feature
    resp = Response(status=200, mimetype='application/json')
    resp.data = json.dumps({"code": 200, "accuracy": 0.0})
    return resp

@app.route("/getVersion", methods=['GET'])
def get_version():
    resp = Response(status=200, mimetype='application/json')
    resp.data = json.dumps({"code": 200, "version": g_version})
    return resp

@app.route("/getParameter", methods=['GET'])
def get_parameter() -> Response:
    """Send parameter to workers

    Returns:
        Response: octet-stream response of serialized json
    
    TODO:
        use encryption to protect model
    """
    global g_lock
    resp = Response(mimetype='application/octet-stream')

    if Net is None:
        resp.data = 500
        resp.data = pickle.dumps({"code": 500, "param": ''})
        return resp

    with g_lock:
        state_dict = g_net.state_dict()
    state_dict_cpu = {}
    for key in state_dict.keys():
        state_dict_cpu[key] = state_dict[key].detach().cpu()

    resp.status = 200
    resp.data = pickle.dumps({"code": 200, "param": state_dict_cpu})
    return resp


@app.route("/putGradient", methods=['POST'])  # app.route does not accept POST actions by default
def put_gradient() -> Response:
    """Get gradient form workers
    Workers post octec_stream data of a serialized dict:
    {
        "id":$WORKER_UNIQUE_ID
        "param":state_dict
    }

    Returns:
        Response: json response of status
    
    TODO:
        Verify id, protect with access_key and secret_key
    """
    global g_gradient_queue, g_policy

    resp = Response(mimetype='application/json')
    data = request.get_data()
    if request.mimetype != 'application/octet-stream':
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    data_dict = pickle.loads(data)
    if not isinstance(data_dict, dict) or "param" not in data_dict.keys():
        resp.status = 409
        resp.data = json.dumps({"code": 409, "msg": "Wrong format"})
        return resp

    gradient_dict: Dict[str, torch.Tensor] = data_dict["param"]
    try:
        g_gradient_queue.put(gradient_dict, block=True, timeout=5)
    except queue.Full:
        resp.status = 503
        resp.data = json.dumps({"code": 503, "msg": "Too busy"})
        return resp

    with g_lock:
        ret: int = run_gradient_descent(g_policy)
    if ret == 500:
        resp.status = 500
        resp.data = json.dumps({"code": 500, "msg": "Error"})
        return resp
    elif ret == 201:
        # Gradient stored, but with no update
        resp.status = 201
        resp.data = json.dumps({"code": 201, "msg": "Error"})
        return resp
    else:
        # A parameter update is triggered
        resp.status = 200
        resp.data = json.dumps({"code": 200, "msg": "OK"})
        return resp


def apply_grad(net: nn.Module, grad_dict: Dict[str, torch.Tensor], gain: float=1.0):
    """Apply gradient to a module

    net.parameters.grad += grad_dict * gain
    Args:
        net (nn.Module): The module to apply on
        grad_dict (Dict[str, torch.Tensor]): The gradient
    """
    net.zero_grad()
    for name, param in net.named_parameters():
        try:
            if param.grad is None:
                param.grad = grad_dict[name] * gain
            else:
                param.grad += grad_dict[name] * gain
        except KeyError as err:
            print(f'[ Warning ] Key {name} does not exist')


def run_gradient_descent(policy: Dict[str, Any]) -> int:
    """Run gradient descent forever

    Args:
        policy (Dict[str, Any]): Defines strategy of train, should at least contains:
        - batch_sz [int]: Gradient descent batch size
        - lr [float]: learning rate
        - save_interval
    """
    global g_gradient_queue, g_net, g_optimizer, g_version
    batch_sz = g_policy["batch_sz"]
    cnt = 0
    grad_list = []
    # If collected enough gradient, run optimizer
    if (g_gradient_queue.qsize() > g_policy["batch_sz"]):
        try:
            while cnt < batch_sz:
                grad_list.append(g_gradient_queue.get_nowait())
                cnt += 1
        except queue.Empty:
            print("Empty Queue")
            return 500
        
        gain = 1 / len(grad_list)
        for curr_grad in grad_list:
            apply_grad(g_net, curr_grad, gain)
        
        print("[ Info ] Optimizer step")
        g_optimizer.step()
        g_optimizer.zero_grad()
        g_version += 1

        return 200
    else:
        return 201
 
def _init_share(queue_maxsize: int=4096):
    """Init share variables

    Args:
        queue_maxsize (int, optional): Shared queue that stores collected gradients. Defaults to 4096.
    """
    global g_gradient_queue
    global g_workers

    g_workers = dict()
    g_gradient_queue = mp.Queue(maxsize=queue_maxsize)

def _init_net(model: nn.Module, 
              pth_to_state_dict: str = None, 
              device: torch.device = torch.device('cpu'), *args, **kwargs):
    """Init neural network

    Args:
        model (nn.Module): Model to be trained
        pth_to_state_dict (str, optional): Load previous checkpoint if needed. Defaults to None.
        device (torch.device, optional): Device to store model. Defaults to torch.device('cpu') (CUDA not supported)

    TODO:
        Support cuda in future
    Returns:
        [type]: [description]
    """
    global g_net, g_lock, g_version


    # Ensure that we get only one handle to the net.
    if g_net is None:
        # construct it once
        g_net = model(*args, **kwargs)
        if pth_to_state_dict is not None:
            g_net.load_state_dict(torch.load(pth_to_state_dict))
        g_net.to(device)
        g_net.train()
        g_version = 0

    return g_net

def _init_optimizer(optim_class: Optimizer, lr: float=1e-6, *args, **kwargs):
    """Init optimizer
    Remark:
        Call this method after _init_net() or you will get error
    """
    global g_optimizer
    if g_optimizer is None:
        g_optimizer = optim_class(g_net.parameters(), lr=lr, *args, **kwargs)
    return g_optimizer



def init(policy: Dict[str, Any]) -> None:
    global g_policy
    g_policy = policy

    _init_share()
    _init_net(Net)
    _init_optimizer(torch.optim.SGD, policy['lr'])
    pass


if __name__ == '__main__':
    DEGUG: bool = True

    parser = argparse.ArgumentParser(description="Parameter-Server HTTP based training")

    parser.add_argument("--master_addr",
                        type=str,
                        default="0.0.0.0",
                        help="""Address of master, will default to 0.0.0.0 if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument("--master_port",
                        type=str,
                        default="29500",
                        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
    parser.add_argument("--batch_sz",
                        type=int,
                        default=4,
                        help="""Batch size of FedSGD""")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="""Batch size of FedSGD""")

    args = parser.parse_args()
    master_addr: str = args.master_addr
    master_port: str = args.master_port

    print(f'[ Info ] Start server at {master_addr}:{master_port}')

    init({"batch_sz":args.batch_sz, "lr":args.lr})
    # Warning: This app does not support multi-process server yet
    # Warning: On Mac OS X, Queue.qsize() is not implemented
    if DEGUG:
        app.run(master_addr, master_port)
    else:
        server = pywsgi.WSGIServer((master_addr, master_port), app)
        server.serve_forever()
