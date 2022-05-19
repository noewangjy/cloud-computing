"""worker_utils.py"""
import requests
import pickle
import torch
import torch.nn as nn
from typing import List, Dict, Union, Any


def get_param_from_remote(apihost: str) -> Union[None, Dict[str, torch.Tensor]]:
    """Get state dict(parameters) from a remote api

    Args:
        apihost (str): url for api

    Returns:
        Union[None, Dict[str, torch.Tensor]]: If succeed, a dictionary of parameters will be obtained
    """
    apihost += '/getParameter'
    model_param = None
    resp = requests.get(apihost)
    if 'application/octet-stream' in resp.headers['Content-Type']:
        resp_dict: Dict[str, Any] = pickle.loads(resp.content)
    else:
        return None

    assert isinstance(resp_dict, dict)
    assert "code" in resp_dict.keys()
    if resp_dict["code"] == 200:
        model_param = resp_dict["param"]

    return model_param


def get_version_from_remote(apihost: str) -> int:
    """Get model parameter version from remote

    Args:
        apihost (str): url for api

    Returns:
        int: version of model parameter
    """
    apihost += '/getVersion'
    resp = requests.get(apihost)
    if resp.status_code != 200:
        return -1
    else:
        resp_dict = resp.json()
        if 'code' in resp_dict and resp_dict['code'] == 200:
            return resp_dict['version']
        else:
            return -1


def get_grad_from_local(net: nn.Module, gain: float = 1.0) -> Dict[str, torch.Tensor]:
    """Bundle parameters of a local nn.Module to Dict

    Args:
        net (nn.Module): [description]

    Returns:
        Dict[str, torch.tensor]: parameter list
    """

    gradient_dict = {}
    module_parameters: List[str, nn.parameter.Parameter] = list(net._named_members(lambda module: module._parameters.items()))
    for name, param in module_parameters:
        gradient_dict[name] = (param.grad.clone().detach() * gain).cpu()
    return gradient_dict


def put_grad_to_remote(apihost: str, grad_dict: Dict[str, torch.Tensor], worker_id: Any = 0):
    """Put gradient to remote parameter server

    Args:
        apihost (str): URL of parameter server api
        grad_dict (Dict[str, torch.Tensor]): dict to put
        worker_id (Any, optional): The id of current worker. Defaults to 0.

    Returns:
        bool: [description]
    """
    apihost += '/putGradient'
    req = requests.post(url=apihost,
                        data=pickle.dumps({
                            "id": worker_id,
                            "param": grad_dict
                        }),
                        headers={"Content-Type": 'application/octet-stream'})
    if req.status_code != 200:
        return False

    if 'code' in req.json().keys() and req.json()['code'] == 200:
        return True
    else:
        return False