## 任务介绍

**手写数字识别**是计算机视觉的一个**经典项目**，因为**手写数字的随机性**，使用传统的计算机视觉技术**难以找到数字共有特征**。在计算机视觉发展的初期，手写数字识别成为一大难题。

从我们之前讲解的视觉任务分类来看，**手写数字识别是典型的分类任务**，输入一张图片进行**十分类**。在现实生活中，手写数字识别也有非常多的应用场景。如下图，我们看到的**邮编的识别**，可以极大地推动产业自动化，使用卷积神经网络实现的**精度甚至可以超越人类**。

本次任务就是想建立一个模型，输入一张手写数字的图片，就能输出一个正确的分类结果。通过这样的一个实战项目，可以很好地帮我们**巩固和理解我们之前讲过的卷积、池化等常用操作**，也可以温习一下深度学习的基本流程。

<div align="center">
<img src="assets/1.png" width="800">
</div>

## 数据准备

手写数字识别有通用的数据集**MNIST**，其中包含已经标注好的几万张手写数字，并且分好了训练集和评价集。如果我们对其中的一张图片进行可视化，可以看到这样的画面：

<div align="center">
<img src="assets/2.png" width="300">
</div>

图像的shape为**（1，28，28）**，是**单通道图**，图像的大小仅为**28*28**，它的**标注为7**。

通常对于一般项目来说，需要自己**手写一个Dataloader来依次加载数据，返回图片和标注，供给训练的接口用于训练**。这里考虑到我们入门的原因，直接使用写好的API。有兴趣的同学可以自己尝试不使用高级API，自己下载好压缩包手写一下Dataloader。

```
train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=10, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)
```

通过上面包装好的API，我们就加载好了**训练集**和**评价集**，可以供训练接口调用。

## 网络搭建

准备好数据之后，第二部也就是**搭建卷积神经网络**，卷积神经网络直接影响着模型的精度，这一步也是**最为关键的一个环节**。本次实战中，我们默认使用**LeNet**。**LeNet**是最早的卷积神经网络之一，**诞生于1998年，在手写数字识别任务中取得了巨大成功**。

<div align="center">
<img src="assets/3.png" width="800">
</div>

它的网络结构也非常简单，基本上为**一个卷积层接着一个池化层**，最后通过两个全连接层输出一个**[1,10]**的矩阵。全连接层我们之前没有介绍过，它通常用于**拟合一些批量数据，比如有很多散点，拟合出一条曲线**。它的结构如下：

<div align="center">
<img src="assets/4.png" width="500">
</div>

也就是说**每一个输出和前面一层的所有参数都相关**，它的数学表达其实就是**乘上一个变换矩阵再加上偏差**，得到输出矩阵。**为什么图像中大量使用卷积层，很少使用全连接层呢？这边留给大家课后自己思考。**

**LeNet**使用**Paddle**复现代码如下：

```
import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        self.fc1 = Linear(in_features=120, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num_classes)
    def forward(self, x):                        #[N,1,28,28] 
        x = self.conv1(x)                        #[N,6,24,24]
        x = F.sigmoid(x)                         #[N,6,24,24]
        x = self.max_pool1(x)                    #[N,6,12,12]
        x = F.sigmoid(x)                         #[N,6,12,12]
        x = self.conv2(x)                        #[N,16,8,8]
        x = self.max_pool2(x)                    #[N,16,4,4]
        x = self.conv3(x)                        #[N,120,1,1]
        x = paddle.reshape(x, [x.shape[0], -1])  #[N,120]
        x = self.fc1(x)                          #[N,64]
        x = F.sigmoid(x)                         #[N,64]
        x = self.fc2(x)                          #[N,10]
        return x
```

Paddle使用**动态图的这种写法非常清晰**，定义一个类体，在初始化函数内写好需要使用的层，**需要特别注意好输入输出的通道数，卷积核的大小这些参数，如果稍不注意就会出现维度上的错误**。在这边定义好之后，我们再写forward函数，**forward函数就是之后我们传入图像后真正执行的运算。**

为了帮助大家理解，再详细解释一下执行流程。首先我们**实例化类体**。

```
model = LeNet(num_classes=10)
```

实例化的时候，类体**自动地执行init()初始化函数**，init()函数里面又实例化了**Conv2D，MaxPool2D**，这些其实都是**类体**，这些类体和LeNet一样，**也有init()和forward函数**，在初始化函数中都进行了相应的实例化。**实例化的过程，并没有真正开始运算**，只是定义好了我想要使用的层。

```
output = model(img)
```

当我再次运行上面的代码后，相当于**调用**了这个**类体**，并且输入了**img**，这时候类体会自动调用**call()函数**，那forward函数为什么会执行呢？原因就在于所有的运算都继承了**paddle.nn.Layer母类**，母类中将**forward函数写在了call()下面**，那么就相当于调用**LeNet**这个类体的时候，**自动调用了forward函数，这时候也就开始了真正的运算过程**。

整个过程希望大家反复推敲，知道彻底理解为止。不难发现，这样的建立网络的形式，可以**不停地嵌套**，这是非常清晰的形式，我们之后讲解复杂模型的时候这样的优势就会体现出来。

## 模型训练

```
# -*- coding: utf-8 -*-
# LeNet 识别手写数字
import imp
import paddle
import numpy as np
import paddle
from model import LeNet
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST


def train(model, opt, train_loader, valid_loader):
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    print('start training ... ')
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            img = data[0]              #[10,1,28,28]
            label = data[1]            #[10,1]
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch+1, batch_id, float(avg_loss.numpy())))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            img = data[0]
            label = data[1] 
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')


model = LeNet(num_classes=10)
EPOCH_NUM = 5
opt = paddle.optimizer.Momentum(learning_rate=0.001, parameters=model.parameters())
train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=10, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)
train(model, opt, train_loader, valid_loader)
```

训练的代码我们根据学过的知识，就非常清晰。从**数据集接口获得数据集**，把图像输入到模型中，模型得到一个预测值，使用**CrossEntropyLoss损失函数**计算预测值和标签真实值的**loss**，将**loss反向发聩给网络参数**，最后使用**优化器**修正参数，降低**loss**。

需要注意的是**CrossEntropyLoss损失函数自带softmax**，分类问题最后都需要一个**softmax激活函数**，**把输出的[1,10]矩阵归到[0,1]，并且10个数的和为1，也就代表了这张图片为0-9的概率**。

## 模型预测

```
import numpy as np
import paddle
from model import LeNet
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor
import paddle.nn.functional as F

valid_loader = MNIST(mode='test', transform=ToTensor())
img = np.array(valid_loader[0][0])

# import matplotlib.pyplot as plt
# plt.imshow(img.squeeze(), cmap='gray')
# plt.show()

model = LeNet(num_classes=10)
model_dict = paddle.load("mnist.pdparams")
model.set_state_dict(model_dict)
model.eval()
x = valid_loader[0][0].reshape((1,1,28,28)).astype('float32')
result = F.softmax(model(x))
print(result.numpy()[0])
```

训练完模型之后，我们需要**加载模型并且预测**，这里就挑选了评价集中的一张图片预测，看一下输出的结果是否正确。

```
model = LeNet(num_classes=10)
model_dict = paddle.load("mnist.pdparams")
model.set_state_dict(model_dict)
```

**我们使用这样的方法加载模型**，最后预测输出：

```
[7.3181213e-06 1.4578840e-05 3.3818762e-04 2.1557527e-04 2.6723552e-05 
 6.7271581e-06 1.3456239e-08 9.9840504e-01 4.1231990e-05 9.4459485e-04]
```

这也就分别代表**0-9的概率**，7的概率高达99.84%，模型输出正确！

## 参考资料

https://www.paddlepaddle.org.cn/tutorials/projectdetail/2227103
