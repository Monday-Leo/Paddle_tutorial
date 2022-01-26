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

class YOLOv3Head(nn.Layer):
    def __init__(self,
                 in_channels=[512, 256, 128],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80):

        super(YOLOv3Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.num_outputs = len(self.anchors)
        self.yolo_outputs = []
        for i in range(len(self.anchors)):
            num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            tip = ConvBNLayer(
                        ch_in=self.in_channels[i],
                        ch_out=self.in_channels[i] * 2,
                        filter_size=3,
                        padding=1)

            conv = nn.Conv2D(
                in_channels=self.in_channels[i]* 2,
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0)

            yolo_output = self.add_sublayer("yolo_head.tip.{}".format(i), tip)
            self.yolo_outputs.append(yolo_output)
            yolo_output = self.add_sublayer("yolo_head.output.{}".format(i), conv)
            self.yolo_outputs.append(yolo_output)

    def forward(self, feats):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[2*i](feat)
            yolo_output = self.yolo_outputs[2*i+1](yolo_output)
            yolo_outputs.append(yolo_output)

        return yolo_outputs

n = YOLOv3Head()