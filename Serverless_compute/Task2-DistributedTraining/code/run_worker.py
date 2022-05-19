"""run_worker.py"""
from typing import Any
import urllib
import os
import time
import tarfile
import json

import torch
import torch.nn as nn

import tqdm
from flask import Flask, request, Response
from gevent import pywsgi

from models import Net
from worker_utils import *

g_net: nn.Module = None
g_train_dataset: torch.utils.data.Dataset
app = Flask(__name__)

g_train_dataset = None


def import_train_dataset_from_url(url: str) -> bool:
    global g_train_dataset
    print(f"[ Info ] Downloading dataset from {url}")
    try:
        urllib.request.urlretrieve(url, 'tmp.tar.gz')
    except Exception:
        if os.path.exists('./tmp.tar.gz'):
            os.remove('./tmp.tar.gz')
        raise Exception

    tar_file = tarfile.open('./tmp.tar.gz')
    tar_file.extractall()

    num_retry: int = 5
    while num_retry > 0:
        try:
            import dataset_dl
            break
        except:
            time.sleep(1)
            num_retry -= 1
            pass

    g_train_dataset = dataset_dl.Dataset
    print(f"[ Info ] Successfully imported dataset: {g_train_dataset.__repr__()}")

    return True


@app.route("/run", methods=['GET', 'POST'])
def run():
    resp = Response(mimetype='application/json')
    try:
        policy = request.get_json()["value"]  # extract real arguments
        print(policy)
    except KeyError:
        print("[ Error ] No argument")
        resp.status = 500
        resp.data = json.dumps({"code": 500, "msg": "No argument"})
        return resp

    try:
        BATCH_SZ_TRAIN: int = int(policy["batch_sz_train"])
        EPOCH_N: int = int(policy["epoch_n"])
        APIHOST: str = policy["apihost"]
        UPDATE_INTV: int = int(policy["update_intv"])
        DATASET_URL: str = policy["dataset_url"]
        DEVICE = torch.device(policy["device"])
    except KeyError:
        print("[ Error ] Wrong parameters")
        resp = Response(status=500)
        resp.data = json.dumps({"code": 500, "msg": "Wrong parameters"})
        return resp

    global g_net, g_train_dataset

    if g_net is None: init()

    assert import_train_dataset_from_url(DATASET_URL)

    g_net.to(DEVICE)
    state_dict = get_param_from_remote(APIHOST)
    g_net.load_state_dict(state_dict)

    train_loader = torch.utils.data.DataLoader(g_train_dataset, batch_size=BATCH_SZ_TRAIN, shuffle=True)

    # Model version
    local_version: int = get_version_from_remote(APIHOST)

    for epoch_idx in range(1, EPOCH_N + 1):
        train_loss_tot: float = 0.0
        train_loss_cnt: int = 0
        with tqdm.tqdm(range(len(train_loader))) as pbar:
            for batch_idx, (stimulis, label) in enumerate(train_loader):
                pred = g_net(stimulis.to(DEVICE))
                # label = torch.nn.functional.one_hot(label, num_classes=10).to(pred.dtype)
                loss = torch.nn.functional.cross_entropy(pred, label.to(DEVICE))
                loss.backward()

                train_loss_tot += float(loss.detach().cpu().numpy())
                train_loss_cnt += 1

                if batch_idx % UPDATE_INTV == 0:
                    local_grad = get_grad_from_local(g_net, 1 / UPDATE_INTV)
                    ret = put_grad_to_remote(APIHOST, local_grad)

                remote_version = get_version_from_remote(APIHOST)
                if remote_version > local_version:
                    local_version = remote_version
                    state_dict = get_param_from_remote(APIHOST)
                    g_net.load_state_dict(state_dict)

                pbar.set_description(f"version: {local_version}loop: {epoch_idx}, avg_loss:{train_loss_tot / train_loss_cnt}")
                pbar.update(1)
    resp.status = 200
    resp.data = json.dumps({"code": 200, "avg_loss": train_loss_tot / train_loss_cnt})
    return resp


@app.route("/init", methods=['GET', 'POST'])
def init():
    global g_net
    g_net = Net()
    g_net.train()
    resp = Response(mimetype='application/json')
    resp.data = json.dumps({"code": 200})
    return resp


if __name__ == '__main__':
    DEGUG: bool = False
    SERVING_PORT: int = 8080

    if DEGUG:
        app.run('0.0.0.0', SERVING_PORT)
    else:
        server = pywsgi.WSGIServer(('0.0.0.0', SERVING_PORT), app)
        server.serve_forever()