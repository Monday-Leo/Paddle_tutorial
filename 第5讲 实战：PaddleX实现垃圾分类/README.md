## PaddleX实现垃圾分类

### B站教程地址

https://www.bilibili.com/video/BV18b4y1J7a6/

### PaddleX的安装

```
pip install paddlex==2.1.0 -i https://mirror.baidu.com/pypi/simple
```

因为**PaddleX依赖于pycocotools**，如果报错：

```
Microsoft Visual C++ 14.0 is required
```

则需要安装相应工具，下载链接如下：

链接：https://pan.baidu.com/s/17pSEX9ZO28_OIPdaeNAe3A 
提取码：xhu4

**如果出现下载预训练模型报错**

```
SSLError("Can't connect to HTTPS URL because the SSL module is not available)
```
需要安装OpenSLL工具，下载链接如下：

链接：[https://pan.baidu.com/s/1Z3F9cIH3-6QrsqnAoK-pBw](https://pan.baidu.com/s/1Z3F9cIH3-6QrsqnAoK-pBw) 
提取码：ucko

### 数据集的准备

数据集下载链接：

链接：https://pan.baidu.com/s/1ZSHQft4eIpYHliKRxZcChQ 
提取码：hce7

本次实战为**图片分类任务**，**数据集结构**如下：

<div align="center">
<img src="assets/1.png" width="400">
</div>

分别为分类的图片文件夹，建议有**一个类别就建一个文件夹**，方便管理。训练集和评价集的标签文件，格式如下：

**./3/933.jpg 3**
**./2/1670.jpg 2**
**./2/2175.jpg 2**
**./1/934.jpg 1**
**./1/1653.jpg 1**

**...**

前面为**图片的相对路径**，后面为**对应的标签类别**。labels.txt存放对应的标签，格式如下：

**有害垃圾**
**可回收垃圾**
**厨房垃圾**
**其他垃圾**

提供文件重命名的代码，**用于每个文件夹里面图片的重命名**

```
import os
 
def rename():
        res = os.listdir('./')
        for a in res:
                i = 0
                flag = os.path.isdir(a)
                if(flag == False):
                        continue
                path=a
                filelist=os.listdir(path)#该文件夹下所有的文件（包括文件夹）
                for files in filelist:#遍历所有文件
                    i=i+1
                    Olddir=os.path.join(path,files);#原来的文件路径                
                    if os.path.isdir(Olddir):#如果是文件夹则跳过
                            continue
                    filename=os.path.splitext(files)[0];#文件名
                    filetype=os.path.splitext(files)[1];#文件扩展名
                    Newdir=os.path.join(path,str(i)+filetype);#新的文件路径
                    os.rename(Olddir,Newdir)#重命名
rename()

```

提供**生成train.txt和eval.txt文件**的代码，分类的**比例为5:1**

```
import os
import random
         
def ReadFileDatas():
    FileNamelist = []
    file = open('train.txt','r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
    #print('len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()
    return FileNamelist
 
def WriteDatasToFile(listInfo):
    file_handle_train=open('train.txt',mode='w')
    file_handle_eval = open("eval.txt",mode='w')
    i = 0
    for idx in range(len(listInfo)):
        str = listInfo[idx]
        #查找最后一个 “_”的位置
        ndex = str.rfind('_')
        #print('ndex = ',ndex)
        #截取字符串
        str_houZhui = str[(ndex+1):]
        #print('str_houZhui = ',str_houZhui)
        str_Result = str  + '\n'           #+ str_houZhui+'\n'
        #print(str_Result)
        if(i%6 != 0):
            file_handle_train.write(str_Result)
        else:
            file_handle_eval.write(str_Result)
        i += 1
    file_handle_train.close()
    file_handle_eval.close()

path = './'
res = os.listdir(path)
print(res)
with open("train.txt","w") as f:
    for i in res:
        if(os.path.isdir(i)):
            path1 = path + i
            res2 = os.listdir(path1)
            for j in res2:
                f.write(path1+"/"+j+" " + i +'\n')

listFileInfo = ReadFileDatas()
#打乱列表中的顺序
random.shuffle(listFileInfo)
WriteDatasToFile(listFileInfo)
```

### 模型训练

借助于**PaddleX**，模型训练变得非常简单，主要分为**数据集定义，数据增强算子定义，模型定义和模型训练**四个步骤：

```
from paddlex import transforms as T
import paddlex as pdx

train_transforms = T.Compose([          #定义训练集的数据增强算子
    T.RandomCrop(crop_size=224),
    T.RandomHorizontalFlip(),
    T.Normalize()])

eval_transforms = T.Compose([			#定义评价集的数据增强算子
    T.ResizeByShort(short_size=256),
    T.CenterCrop(crop_size=224),
    T.Normalize()
])

train_dataset = pdx.datasets.ImageNet(		#定义训练集
    data_dir='rubbish',
    file_list='rubbish/train.txt',
    label_list='rubbish/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(		#定义评价集
    data_dir='rubbish',
    file_list='rubbish/eval.txt',
    label_list='rubbish/labels.txt',
    transforms=eval_transforms)

num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_small(num_classes=num_classes)		#定义分类模型

model.train(num_epochs=10,										#模型训练
            train_dataset=train_dataset,
            train_batch_size=64,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_dir='output/mobilenetv3_small',
            use_vdl=True)
```

具体参数的含义可以参照**PaddleX的[Github](https://github.com/PaddlePaddle/PaddleX)文档**，在B站视频中也做了详细的讲解。

### 模型的预测

```
import paddlex as pdx
model = pdx.load_model('output/mobilenetv3_small/best_model')
result = model.predict('188.jpg')
print("Predict Result: ", result)
```

可以观察输出的结果是否正确。

### 模型训练的可视化

```
visualdl --logdir output/mobilenetv3_small --port 8001
```

打开浏览器输出网址，可以看到训练的**各个参数曲线**。