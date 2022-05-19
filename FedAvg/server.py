import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from models import Mnist_2NN, Mnist_CNN
from client import ClientGroup, SingleClient
# from model.WideResmodel import WideResmodel




def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    # 客户端的数量
    parser.add_argument('-nc', '--num_of_clients', type=int, default=1000, help='numer of the clients')
    # 随机挑选的客户端的数量
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
    # 训练次数(客户端更新次数)
    parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
    # batchsize大小
    parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
    # 模型名称
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
    # 学习率
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-dataset',"--dataset",type=str,default="mnist",help="需要训练的数据集")
    # 模型验证频率（通信频率）
    parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    #n um_comm 表示通信次数，此处设置为1k
    parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=bool, default=False, help='the way to allocate data to clients')



    args = parser.parse_args()
    args = args.__dict__


    #-----------------------文件保存-----------------------#
    # 创建结果文件夹
    test_mkdir("./result")
    test_txt = open("result/test_accuracy.txt", mode="a")
    global_parameters_txt = open("result/global_parameters.txt",mode="a",encoding="utf-8")
    #----------------------------------------------------#
    # 创建最后的结果
    test_mkdir(args['save_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = None
    # 初始化模型
    # mnist_2nn
    if args['model_name'] == 'mnist_2nn':
        model = Mnist_2NN()
    # mnist_cnn
    elif args['model_name'] == 'mnist_cnn':
        model = Mnist_CNN()
    else:
        raise ValueError("model_name must be in ['mnist_2nn', 'mnist_cnn']")

    ## 如果有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # 将Tenor 张量 放在 GPU上
    model = model.to(device)

    '''
        回头直接放在模型内部
    '''
    # 定义损失函数
    loss_fct = F.cross_entropy
    # 优化算法的，随机梯度下降法
    # 使用Adam下降法
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

    ## 创建Clients群
    '''
        创建Clients群100个
        
        得到Mnist数据
        
        一共有60000个样本
        100个客户端
        IID：
            我们首先将数据集打乱，然后为每个Client分配600个样本。
        Non-IID：
            我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
            然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
            注： 我觉得着并不是真正意义上的Non—IID
    '''
    myClients = ClientGroup('mnist', args['batchsize'], args['IID'], args['num_of_clients'], device)
    testDataLoader = myClients.test_data_loader

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    # 每次随机选取10个Clients
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 得到全局的参数
    global_parameters = {}
    # model.state_dict()  # 获取模型参数以共享

    # 得到每一层中全连接层中的名称fc1.weight
    # 以及权重weights(tenor)
    # 得到网络每一层上
    for key, var in model.state_dict().items():
        # print("key:"+str(key)+",var:"+str(var))
        print("张量的维度:"+str(var.shape))
        print("张量的Size"+str(var.size()))
        global_parameters[key] = var.clone()



    # num_comm 表示通信次数，此处设置为1k
    # 通讯次数一共1000次
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        # 对随机选的将100个客户端进行随机排序
        order = np.random.permutation(args['num_of_clients'])
        # 生成个客户端
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]


        print("Client Group: "+str(clients_in_comm))

        sum_parameters = None
        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        # 返回每个Client更新后的参数

        for client in tqdm(clients_in_comm):
            # 获取当前Client训练得到的参数
            # 这一行代码表示Client端的训练函数，我们详细展开：
            # local_parameters 得到客户端的局部变量
            local_parameters = myClients.clients_set[client].local_update(args['epoch'], args['batchsize'], model,
                                                                         loss_fct, optimizer, global_parameters)
            # 对所有的Client返回的参数累加（最后取平均值）
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # 取平均值，得到本次通信中Server得到的更新后的模型参数
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        #test_txt.write("communicate round " + str(i + 1) + str('accuracy: {}'.format(sum_accu / num)) + "\n")

        '''
            训练结束之后，我们要通过测试集来验证方法的泛化性，
            注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的
            还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        '''
        
        #  加载Server在最后得到的模型参数
        model.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(device), label.to(device)
            preds = model(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print('accuracy: {}\n'.format(sum_accu / num))

        test_txt.write("communicate round "+str(i+1)+"  ")
        test_txt.write('accuracy: '+str(float(sum_accu / num))+"\n")
        #test_txt.close()

        if (i + 1) % args['save_freq'] == 0:
            torch.save(model, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

    test_txt.close()