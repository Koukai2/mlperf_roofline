import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import torchvision
 
def imshow(img):
    img = img / 2 + 0.5     # unnormalize，因为 ToTensor() 把数据归一化了
    npimg = img.numpy()     # 把 tensor 转换成 numpy 数组
    plt.imshow(np.transpose(npimg, (1, 2, 0)))      # 把通道维度放到最后，因为 matplotlib 需要的是 (H, W, C)
    plt.show()            # 显示图片
 
dataiter = iter(train_dataloader)    # 创建一个迭代器
images, labels = next(dataiter)     # 返回一个 batch 的数据
imshow(torchvision.utils.make_grid(images))    # 把这个 batch 的图像拼成一个网格
print(' '.join('%5s' % training_data.classes[labels[j]] for j in range(batch_size)))    # 打印这个 batch 的标签
 
