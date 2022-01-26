from darknet import DarkNet
from neck import YOLOv3FPN
from head import YOLOv3Head
import paddle

inputs = paddle.rand((1,3,416,416))
backbone = DarkNet()
neck = YOLOv3FPN()
head = YOLOv3Head(num_classes=20)

outs = backbone(inputs)
for out in outs:
    print(out.shape)

outs = neck(outs)
for out in outs:
    print(out.shape)

outs = head(outs)
for out in outs:
    print(out.shape)