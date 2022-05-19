import torch
import torchvision

_raw_dataset = torchvision.datasets.MNIST(
        './dataset_dl/data/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ]))

dataset = torch.utils.data.Subset(_raw_dataset, range(50000, 60000))