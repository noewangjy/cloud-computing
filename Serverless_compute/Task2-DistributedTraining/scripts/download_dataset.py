if __name__ == '__main__':
    import torchvision
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    