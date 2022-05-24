# 云计算

本仓库是上海交通大学ICE6405P-云计算（2021年秋）的项目代码及项目报告。

课程网站请访问[这里](https://tsingz0.github.io/ICE6405P-260-M01/).

作者：[Steve Wang](mailto:steve_wang@sjtu.edu.cn)

## 目录

本仓库的项目如下:

1. [基于KVM的网络虚拟化和虚拟机热迁移](Virtualization)
2. [基于无服务计算框架的分布式训练及机器学习应用](Serverless_compute)
   - 任务1: 基于无服务器计算框架的机器学习应用
   - 任务2：基于无服务器计算框架的分布式训练
3. [基于云计算平台的联邦学习部署和研究](FedAvg)

## 项目简介

### 1. 基于KVM的网络虚拟化和虚拟机热迁移

在本任务中，我们需要完成以下要求：

1. 基于QEMU/KVM创建一个使用`virtio-net`的虚拟机：
    - 虚拟机安装CentOS操作系统；
    - 虚拟机可以通过`virtio-net` 设备访问外部网络；

2. 高性能的虚拟网络是云计算核心之一：
    - 基于DPDK和OVS，部署`vhost-user` 方案；
    - 评测`vhost-user` 方案和`virtio-net`方案；

3. 向高性能+多功能的方向优化：
    - 配置`vhost-user` 和QEMU启动参数，研究网络设备的多队列特性；
    - 在`vhost-user` 的基础上进行虚拟机热迁移；

本实验的计算资源为：

- 计算平台：
    - `OS`: macOS Catalina 10.15.7
    - `Memory`: 32GB
    - `Processor`: Intel Core i9-9880H @ 2.3GHz * 16
    - `Platform`: VMware Fusion Pro 11.5.0

- 虚拟机环境：
    - `OS`: Ubuntu 20.04LTS
    - `Memory`: 16GB
    - `Disk Capacity`: 110GB
    - `Processor`: Intel Core i9-9880H @ 2.3GHz * 8

---





### 2. 基于无服务计算框架的分布式训练及机器学习应用

#### 任务1: 基于无服务器计算框架的机器学习应用

首先，我们要选择合适的无服务器计算框架，在此项目中，我们有如下三个选择：

1. AWS Lamda. AWS Lambda是最为广泛应用的无服务器计算框架，AWS为学生提供了免费的使用额度，使我们可以在AWS Lambda上部署我们的计算函数。
2. OpenWhisk. OpenWhisk是一个开源的无服务器计算框架，我们需要在本地部署 OpenWhisk 框架，并提交框架部署方法记录。
3. 其他**开源**无服务器计算框架。

在本项目中，我们选择使用开源的无服务器计算框架OpenWhisk。Openwhisk是属于Apache基金会的开源Faas计算平台，由IBM在2016年公布并贡献给开源社区。IBM Cloud本身也提供完全托管的OpenWhisk Faas服务IBM Cloud Function。从业务逻辑来看，OpenWhisk同AWS Lambda一样，为用户提供基于事件驱动的无状态的计算模型，并直接支持多种编程语言。

本章节使用的ML预测性任务为基于CNN的MNIST手写数字辨识项目，我们从`GitHub` 上找到了一个使用`PyTorch`在MNIST数据集上训练好的CNN模型，以及相关的部署代码。由于OpenWhisk框架的限制，所有函数的输入和输出都必须为`.json` 文件，这使得直接传入图片变得有些困难。该代码将ML模型包装成一个WebApp，以便于OpenWhisk的函数在docker容器中可以仅通过URL来获取图片。

在本实验中，我们在一个开源的Docker容器中，下载已经训练好的开源的模型检查点，使用HTTP Server将其加载到模型，并使用OpenWhisk框架进行预测任务。我们的工作流程如下：

1. 从开源项目中搭建Docker镜像；
2. 在Docker中运行基于flask的WebApp；
3. 创建OpenWhisk Action；
4. 启动HTTP Server；
5. Invoke刚才定义的OpenWhisk Action，传入预测图片的URL；
6. 获得预测结果。

本实验的计算资源为：

- 计算平台：
    - `OS`: macOS Catalina 10.15.7
    - `Memory`: 32GB
    - `Processor`: Intel Core i9-9880H @ 2.3GHz * 16
    - `Platform`: VMware Fusion Pro 11.5.0

- 虚拟机环境：
    - `OS`: Ubuntu 20.04LTS
    - `Memory`: 16GB
    - `Disk Capacity`: 110GB
    - `Processor`: Intel Core i9-9880H @ 2.3GHz * 8


#### 任务2：基于无服务器计算框架的分布式训练

在本实验中，我们需要自定义分布式机器学习训练任务，在任务1中选择的无服务器计算框架部署训练任务，并进行分布式训练。在任务1中，我们选择开源的无服务器计算框架OpenWhisk，并选择基于PyTorch实现的MINIST数据集手写数字辨识任务。在本任务中，我们依然选择MNIST手写数字辨识任务，并基于OpenWhisk框架对任务1中的LeNet5模型进行分布式训练。

本实验的计算资源为：

- 计算平台：
    - `OS`: macOS Catalina 10.15.7
    - `Memory`: 32GB
    - `Processor`: Intel Core i9-9880H @ 2.3GHz * 16
    - `Platform`: VMware Fusion Pro 11.5.0

- 虚拟机环境：
    - `OS`: Ubuntu 20.04LTS
    - `Memory`: 16GB
    - `Disk Capacity`: 110GB
    - `Processor`: Intel Core i9-9880H @ 2.3GHz * 8

---

### 3. 基于云计算平台的联邦学习部署和研究

在本项目中，我们将参考[这篇论文](https://arxiv.org/pdf/1602.05629.pdf) ，根据其中的`Algorithm 1` 来实现`FedAvg`算法。我们使用[Conda](https://www.anaconda.com/products/individual) 环境中的Python + [PyTorch](https://pytorch.org/) 进行编程，并在[上海交通大学云计算平台](https://home.jcloud.sjtu.edu.cn/) 部署模型进行训练。 我们选择基于PyTorch实现的MNIST手写数字辨识任务，使用`FedAvg`算法来实现在多个客户端的联邦学习。

在本项目中，我们手动实现非独立同分布（Non-IID）的MNIST数据集，并探索不同数量的客户机对算法准确率的影响。我们的模型部署在上海交通大学云计算平台上，所有的模型训练和参数更新都是基于CPU的。在初步的实验中，我们使用循环算法来实现多客户机的训练，在后续的实验中，我们使用Python的多进程模块来进一步实现客户机的并行计算。

本实验的计算资源为：

- 上海交通大学云计算平台

    - A服务器* 1：
        - `OS`: Ubuntu 20.04 LTS
        - `Memory`: 128GB
        - `Processor`: Intel Xeon Processor @ 2.4GHz * 64
        - `Disk Capacity`: 500GB
        - `GPU`: None

    - B服务器* 4：
        - `OS`: Ubuntu 20.04 LTS
        - `Memory`: 64GB
        - `Processor`: Intel Xeon Processor @ 2.4GHz * 32
        - `Disk Capacity`: 500GB
        - `GPU`: None

- 实验环境：
    - `Anaconda3-2021.11-Linux-x86_64`
    - `Python 3.8.12`
    - `PyTorch 1.10.1`

## 致谢

- 感谢为本仓库提供教程和在实验过程中提供指导和帮助的 [davidliyutong](https://github.com/davidliyutong/)

## 参考资料

- [GitHub - davidliyutonf/ICE6405P-260-M01](https://github.com/davidliyutong/ICE6405P-260-M01/)
