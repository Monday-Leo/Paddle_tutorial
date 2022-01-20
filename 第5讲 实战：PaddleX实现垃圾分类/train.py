from paddlex import transforms as T
import paddlex as pdx

train_transforms = T.Compose([
    T.RandomCrop(crop_size=224),
    T.RandomHorizontalFlip(),
    T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256),
    T.CenterCrop(crop_size=224),
    T.Normalize()
])

train_dataset = pdx.datasets.ImageNet(
    data_dir='rubbish',
    file_list='rubbish/train.txt',
    label_list='rubbish/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='rubbish',
    file_list='rubbish/eval.txt',
    label_list='rubbish/labels.txt',
    transforms=eval_transforms)

num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_small(num_classes=num_classes)

model.train(num_epochs=10,
            train_dataset=train_dataset,
            train_batch_size=64,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_dir='output/mobilenetv3_small',
            use_vdl=True)