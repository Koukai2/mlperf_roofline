# 构建神经网络
import torch.nn as nn
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
 
# 定义网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # flatten 层把尺寸为 (n, m, x, y) 的输入转换成 (n, m * x * y) 的输出, n 是 batch_size
        self.flatten = nn.Flatten()
        # Sequential 是一个有序的容器，神经网络层将按照在传入构造器层对象的顺序依次被添加到计算图中执行
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 输入层，Linear 是全连接层
            nn.ReLU(), # 激活函数
            nn.Linear(512, 512), # 隐藏层
            nn.ReLU(), # 激活函数
            nn.Linear(512, 10) # 输出层
        )
 
    def forward(self, x):
        # x = self.flatten(x)
        # x = self.linear_relu_stack(x)
        # return x
        return self.linear_relu_stack(self.flatten(x))
    
model = NeuralNetwork().to(device) # 实例化网络，把它移动到 GPU 上（如果有的话）
print(model)



