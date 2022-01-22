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