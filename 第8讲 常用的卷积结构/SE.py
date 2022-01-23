from paddle.nn import Linear,AdaptiveAvgPool2D
import paddle
import paddle.nn as nn 
import paddle.nn.functional as F

class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio):
        super(SELayer, self).__init__()
        self.pool2d_gap = AdaptiveAvgPool2D(1)
        self._num_channels = num_channels
        self.squeeze = Linear(num_channels,int(num_channels / reduction_ratio))
        self.excitation = Linear(int(num_channels / reduction_ratio),num_filters)

    def forward(self, input):
        pool = self.pool2d_gap(input)                           #[1,256,1,1]          
        pool = paddle.squeeze(pool, axis=[2, 3])                #[1,256]
        squeeze = self.squeeze(pool)                            #[1,16]
        squeeze = F.relu(squeeze)                               #[1,16]
        excitation = self.excitation(squeeze)                   #[1,256]
        excitation = F.sigmoid(excitation)                      #[1,256]
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])  #[1,256,1,1]
        out = input * excitation                                #[1,256,64,64]
        return out

# Basic = SELayer(num_channels=256,num_filters=256,reduction_ratio=16)
# params_info  = paddle.summary(Basic,(1,256,64,64))
# print(params_info)