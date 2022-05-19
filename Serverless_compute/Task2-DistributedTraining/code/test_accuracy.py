import torch
from run_worker import get_param_from_remote
from models import Net

def test(device: torch.device = torch.device('cpu')) -> None:
    BATCH_SZ_TEST: int = 16
    APIHOST: str = 'http://localhost:29500'


    import torchvision
    import tqdm

    net = Net()
    state_dict = get_param_from_remote(APIHOST)
    net.load_state_dict(state_dict)
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
    test()