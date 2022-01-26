import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

def batch_norm(ch,
               norm_type='bn',
               norm_decay=0.,
               freeze_norm=False,
               initializer=None,
               data_format='NCHW'):
    if norm_type == 'sync_bn':
        batch_norm = nn.SyncBatchNorm
    else:
        batch_norm = nn.BatchNorm2D

    norm_lr = 0. if freeze_norm else 1.
    weight_attr = ParamAttr(
        initializer=initializer,
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)
    bias_attr = ParamAttr(
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)

    norm_layer = batch_norm(
        ch,
        weight_attr=weight_attr,
        bias_attr=bias_attr,
        data_format=data_format)

    norm_params = norm_layer.parameters()
    if freeze_norm:
        for param in norm_params:
            param.stop_gradient = True

    return norm_layer

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 norm_type='bn',
                 norm_decay=0.,
                 act="leaky",
                 freeze_norm=False,
                 data_format='NCHW',
                 name=''):

        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            data_format=data_format,
            bias_attr=False)
        self.batch_norm = batch_norm(
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(out, 0.1)
        return out

class YoloDetBlock(nn.Layer):
    def __init__(self,
                 ch_in,
                 channel,
                 name):

        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel

        conv_def = [
            [ch_in, channel, 1],
            [channel, channel * 2, 3],
            [channel * 2, channel, 1],
            [channel, channel * 2, 3],
            [channel * 2, channel, 1],
        ]

        self.conv_module = nn.Sequential()
        for idx, (ch_in, ch_out, filter_size) in enumerate(conv_def):
            self.conv_module.add_sublayer(
                name+"{}".format(idx),
                ConvBNLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=filter_size,
                    padding=(filter_size - 1) // 2))

    def forward(self, inputs):
        route = self.conv_module(inputs)
        return route

class YOLOv3FPN(nn.Layer):

    def __init__(self,in_channels=[256, 512, 1024]):

        super(YOLOv3FPN, self).__init__()
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        self._out_channels = []
        self.yolo_blocks = []
        self.routes = []
        for i in range(self.num_blocks):
            name = 'yolo_block.{}'.format(i)
            in_channel = in_channels[-i - 1]
            if i > 0:
                in_channel += 512 // (2**i)
            yolo_block = self.add_sublayer(
                name,
                YoloDetBlock(
                    in_channel,
                    channel=512 // (2**i),
                    name=name))

            self.yolo_blocks.append(yolo_block)

            if i < self.num_blocks - 1:
                name = 'yolo_transition.{}'.format(i)
                route = self.add_sublayer(
                    name,
                    ConvBNLayer(
                        ch_in=512 // (2**i),
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        name=name))
                self.routes.append(route)

    def forward(self, blocks):
        blocks = blocks[::-1]
        yolo_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route = self.yolo_blocks[i](block)
            yolo_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(route, scale_factor=2.)
        else:
            return yolo_feats


