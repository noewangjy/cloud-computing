# Dataset for deployment

This folder contains splited MNIST dataset

```text
./
├── README.md
├── dataset_dl_1.tar.gz
├── dataset_dl_2.tar.gz
├── dataset_dl_3.tar.gz
├── dataset_dl_4.tar.gz
├── dataset_dl_5.tar.gz
└── dataset_dl_6.tar.gz
```

Each archive is organized as follows

```text
./
├── __dataset__.py
├── __init__.py
├── __pycache__
│   ├── __dataset__.cpython-38.pyc
│   └── __init__.cpython-38.pyc
└── data
    └── MNIST
        └── raw
            ├── t10k-images-idx3-ubyte
            ├── t10k-images-idx3-ubyte.gz
            ├── t10k-labels-idx1-ubyte
            ├── t10k-labels-idx1-ubyte.gz
            ├── train-images-idx3-ubyte
            ├── train-images-idx3-ubyte.gz
            ├── train-labels-idx1-ubyte
            └── train-labels-idx1-ubyte.gz
```
