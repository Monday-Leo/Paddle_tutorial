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
    def forward(self, x):                        #[10,1,28,28] 
        x = self.conv1(x)                        #[10,6,24,24]
        x = F.sigmoid(x)                         #[10,6,24,24]
        x = self.max_pool1(x)                    #[10,6,12,12]
        x = F.sigmoid(x)                         #[10,6,12,12]
        x = self.conv2(x)                        #[10,16,8,8]
        x = self.max_pool2(x)                    #[10,16,4,4]
        x = self.conv3(x)                        #[10,120,1,1]
        x = paddle.reshape(x, [x.shape[0], -1])  #[10,120]
        x = self.fc1(x)                          #[10,64]
        x = F.sigmoid(x)                         #[10,64]
        x = self.fc2(x)                          #[10,10]
        return x