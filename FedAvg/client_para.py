import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import get_dataset


class SingleClient(object):
    def __init__(self, train_dataset, device, name):
        self.train_dataset = train_dataset
        self.device = device
        self.name = name
        self.train_dataloader = None
        self.local_params = None

    
    def local_update(self, epoch, batch_size, model, loss_fct, optimizer, global_params):
        '''
            param: epoch: local epoch
            param: batch_size: local batch size
            param: model: the shared model to train
            param: loss_fct: loss function
            param: optimizer: the optimizer
            param: global_params: the global parameters in this round
            
            return: local model parameters after training  
        '''

        # Update the newest global parameters
        model.load_state_dict(global_params, strict=True)

        # Load local dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        # Let's train
        for epoch_idx in range(epoch):
            for data, labels in self.train_dataloader:
                predictions = model(data.to(self.device))
                loss = loss_fct(predictions, labels.to(self.device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model.state_dict()


class ClientGroup(object):
    '''
        This class is to set a group of clients
    '''
    def __init__(self, dataset_name, batch_size, is_iid, n_clients, device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.is_iid = is_iid
        self.num_of_clients = n_clients
        self.device = device
        self.clients_set = {}

        self.test_data_loader = None

        self.dataset_allocator()

    def dataset_allocator(self):


        dataset = get_dataset(self.dataset_name, self.is_iid)

        test_data = torch.tensor(dataset.test_data)
        test_label = torch.argmax(torch.tensor(dataset.test_label), dim=1)

        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=self.batch_size, shuffle=False)

        train_data = dataset.train_data
        train_label = dataset.train_label

        # Number of training samples for each client
        shard_size = dataset.train_data_size // self.num_of_clients // 2

        # Shuffle the shards
        shards_id = np.random.permutation(dataset.train_data_size // shard_size)

        for client_idx in range(self.num_of_clients):

            shards_id1 = shards_id[client_idx * 2]
            shards_id2 = shards_id[client_idx * 2 + 1]

            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)

            # 创建一个客户端
            current_client = SingleClient(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.device, "client" + str(client_idx))
            # 为每一个clients 设置一个名字
            # client10
            self.clients_set['client{}'.format(client_idx)] = current_client