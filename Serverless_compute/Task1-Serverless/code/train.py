import torch
from model import LeNet5


def train():
    BATCH_SZ_TRAIN: int = 32
    LR: float = 1e-4
    EPOCH_N: int = 20

    import torchvision
    import tqdm

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                               batch_size=BATCH_SZ_TRAIN,
                                               shuffle=True)

    net = LeNet5()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()

    for epoch_idx in range(1, EPOCH_N + 1):
        train_loss_tot: float = 0.0
        train_loss_cnt: int = 0
        with tqdm.tqdm(range(len(train_loader))) as pbar:
            for batch_idx, (stimulis, label) in enumerate(train_loader):
                optimizer.zero_grad()
                pred = net(stimulis)
                # label = torch.nn.functional.one_hot(label, num_classes=10).to(pred.dtype)
                loss = torch.nn.functional.cross_entropy(pred, label)
                loss.backward()
                optimizer.step()
                train_loss_tot += float(loss.detach().cpu().numpy())
                train_loss_cnt += 1
                pbar.set_description("loop: {}, avg_loss:{}".format(
                    epoch_idx, train_loss_tot / train_loss_cnt))
                pbar.update(1)

    torch.save(net.state_dict(), './model.pth')


def test(
    pth_to_state_dict: str,
    device: torch.device = torch.device('cpu')) -> None:
    BATCH_SZ_TEST: int = 16

    import torchvision
    import tqdm

    net = LeNet5()
    net.load_state_dict(torch.load(pth_to_state_dict))
    net.to(device)
    net.eval()

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data/',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                              batch_size=BATCH_SZ_TEST,
                                              shuffle=True)

    acc_cnt: int = 0
    tot_cnt: int = 1e-5

    with tqdm.tqdm(range(len(test_loader))) as pbar:
        for batch_idx, (stimulis, label) in enumerate(test_loader):
            pred = net(stimulis)
            # label = torch.nn.functional.one_hot(label, num_classes=10).to(pred.dtype)
            pred_decoded = torch.argmax(pred, dim=1)
            acc_cnt += (pred_decoded == label).sum().detach().cpu().numpy()
            tot_cnt += pred_decoded.size(0)
            pbar.set_description("acc:{}".format(acc_cnt / tot_cnt))
            pbar.update(1)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        test('./model.pth')
        exit()

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test('./model.pth')
    else:
        raise NotImplementedError
