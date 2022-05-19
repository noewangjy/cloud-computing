# An example of how to organize dataset

```text
./
├── README.md
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

The data folder stores raw data, therefore not tracked by version control.
