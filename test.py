import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 本教程以 FashionMNIST 数据集为例子，这个数据集包含了 10 个类别，共计 7 万张灰度图像。
# 我们使用其中 60,000 张图像训练网络，使用另外 10,000 张图像评估训练得到的网络
# PyTorch 提供了专门的模块加载 FashionMNIST 数据集，见 torchvision.datasets。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
 
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 使用 DataLoader 来帮助我们把数据分成 batch_size=64 的小块，此外还可以打乱数据。
batch_size = 64
 
# 创建训练数据集的数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
 
print(type(test_dataloader))
 
for X, y in test_dataloader:
    # dataloader 里面的数据是以 batch_size 为单位的<tensor, tensor>，其中第一个 tensor 是图像，第二个 tensor 是标签。
    print(type(X), type(y))
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
