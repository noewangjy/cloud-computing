import torch
from typing import Callable, Union


class LeNet5(torch.nn.Module):
    INPUT_HEIGHT: int = 28
    INPUT_WIDTH: int = 28
    def __init__(self, filter_fn: Union[None, Callable[[torch.Tensor], torch.Tensor]]=None):
        super().__init__()

        # Defining LeNet layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), padding=(2,2))
        self.maxp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5))
        self.maxp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.d1 = torch.nn.Linear(16*5*5, 120)
        self.d2 = torch.nn.Linear(120, 84)
        self.d3 = torch.nn.Linear(84, 10)

        self.filter_fn = filter_fn

    def forward(self, x):
        if self.filter_fn is not None:
            x = self.filter_fn(x)
        
        batch_sz = x.shape[0]

        x = torch.relu(self.conv1(x))
        x = self.maxp1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxp2(x)
        x = x.view(batch_sz, -1)
        x = torch.relu(self.d1(x))
        x = torch.relu(self.d2(x))
        x = self.d3(x)
        # x = torch.softmax(self.d3(x), 0)
        return x

# Expose LeNet5 as Net
Net = LeNet5