import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm
from SE import SELayer

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None
                 ):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups)

        self._batch_norm = BatchNorm(num_filters)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if(self.act=="relu"):
            y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu")

        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            act=None)

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        y = paddle.add(x=inputs, y=conv1)
        y = F.relu(y)
        return y

class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 use_se = True,
                 reduction_ratio = 16):
        super(BottleneckBlock, self).__init__()
        self.use_se = use_se

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu")

        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu")

        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)
        if use_se:
            self.scale = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=reduction_ratio)

    def forward(self, inputs):
        y = self.conv0(inputs)
        y = self.conv1(y)
        y = self.conv2(y)
        if(self.use_se):
            y = self.scale(y)
        y = paddle.add(inputs, y)
        y = F.relu(y)
        return y

# Basic = BasicBlock(num_channels=256,num_filters=256,stride=1)
# params_info  = paddle.summary(Basic,(1,256,64,64))
# print(params_info)
# Basic = BottleneckBlock(num_channels=256,num_filters=64,stride=1,use_se=False)
# params_info  = paddle.summary(Basic,(1,256,64,64))
# print(params_info)
