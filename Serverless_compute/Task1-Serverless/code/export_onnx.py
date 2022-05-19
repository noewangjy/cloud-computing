import torch
from model import LeNet5
import onnx
import numpy as np

def pytorch2onnx(pth_to_state_dict: str) -> None:
    import torchvision
    import tqdm

    net = LeNet5()
    net.load_state_dict(torch.load(pth_to_state_dict))
    net.to(torch.device('cpu'))
    net.eval()
    
    dummy_input = torch.randn((1, 1, 28, 28))
    torch.onnx.export(net,dummy_input, 'model.onnx',input_names=['input_0'],output_names=['output_0'], dynamic_axes={'input_0':[0],'output_0':[0]} )

    print("Finished generating ONNX")


if __name__ == '__main__':
    pytorch2onnx('./model.pth')