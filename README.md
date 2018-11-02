# Road-Lane-Instance-Segmentation-PyTorch
Road lane instance segmentation with PyTorch, SegNet and discriminative loss.  
Trained from tuSimple dataset.

### Dataset
Downloads: [tuSimple dataset](https://github.com/TuSimple/tusimple-benchmark/wiki)
#### Load Dataset
```python
train_path = '/data/tuSimple/train_set/'
train_dataset = tuSimpleDataset(train_path, size=SIZE)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
```

### Result
![png](output_0.png)

### Model
#### SegNet with Discriminative Loss using pre-trained vgg16 encoder
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
       BatchNorm2d-2         [-1, 64, 224, 224]             128
              ReLU-3         [-1, 64, 224, 224]               0
        ConvBnRelu-4         [-1, 64, 224, 224]               0
            Conv2d-5         [-1, 64, 224, 224]          36,928
       BatchNorm2d-6         [-1, 64, 224, 224]             128
              ReLU-7         [-1, 64, 224, 224]               0
        ConvBnRelu-8         [-1, 64, 224, 224]               0
            Conv2d-9        [-1, 128, 112, 112]          73,856
      BatchNorm2d-10        [-1, 128, 112, 112]             256
             ReLU-11        [-1, 128, 112, 112]               0
       ConvBnRelu-12        [-1, 128, 112, 112]               0
           Conv2d-13        [-1, 128, 112, 112]         147,584
      BatchNorm2d-14        [-1, 128, 112, 112]             256
             ReLU-15        [-1, 128, 112, 112]               0
       ConvBnRelu-16        [-1, 128, 112, 112]               0
           Conv2d-17          [-1, 256, 56, 56]         295,168
      BatchNorm2d-18          [-1, 256, 56, 56]             512
             ReLU-19          [-1, 256, 56, 56]               0
       ConvBnRelu-20          [-1, 256, 56, 56]               0
           Conv2d-21          [-1, 256, 56, 56]         590,080
      BatchNorm2d-22          [-1, 256, 56, 56]             512
             ReLU-23          [-1, 256, 56, 56]               0
       ConvBnRelu-24          [-1, 256, 56, 56]               0
           Conv2d-25          [-1, 256, 56, 56]         590,080
      BatchNorm2d-26          [-1, 256, 56, 56]             512
             ReLU-27          [-1, 256, 56, 56]               0
       ConvBnRelu-28          [-1, 256, 56, 56]               0
           Conv2d-29          [-1, 512, 28, 28]       1,180,160
      BatchNorm2d-30          [-1, 512, 28, 28]           1,024
             ReLU-31          [-1, 512, 28, 28]               0
       ConvBnRelu-32          [-1, 512, 28, 28]               0
           Conv2d-33          [-1, 512, 28, 28]       2,359,808
      BatchNorm2d-34          [-1, 512, 28, 28]           1,024
             ReLU-35          [-1, 512, 28, 28]               0
       ConvBnRelu-36          [-1, 512, 28, 28]               0
           Conv2d-37          [-1, 512, 28, 28]       2,359,808
      BatchNorm2d-38          [-1, 512, 28, 28]           1,024
             ReLU-39          [-1, 512, 28, 28]               0
       ConvBnRelu-40          [-1, 512, 28, 28]               0
           Conv2d-41          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-42          [-1, 512, 14, 14]           1,024
             ReLU-43          [-1, 512, 14, 14]               0
       ConvBnRelu-44          [-1, 512, 14, 14]               0
           Conv2d-45          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-46          [-1, 512, 14, 14]           1,024
             ReLU-47          [-1, 512, 14, 14]               0
       ConvBnRelu-48          [-1, 512, 14, 14]               0
           Conv2d-49          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-50          [-1, 512, 14, 14]           1,024
             ReLU-51          [-1, 512, 14, 14]               0
       ConvBnRelu-52          [-1, 512, 14, 14]               0
           Conv2d-53          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-54          [-1, 512, 14, 14]           1,024
             ReLU-55          [-1, 512, 14, 14]               0
       ConvBnRelu-56          [-1, 512, 14, 14]               0
           Conv2d-57          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-58          [-1, 512, 14, 14]           1,024
             ReLU-59          [-1, 512, 14, 14]               0
       ConvBnRelu-60          [-1, 512, 14, 14]               0
           Conv2d-61          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-62          [-1, 512, 14, 14]           1,024
             ReLU-63          [-1, 512, 14, 14]               0
       ConvBnRelu-64          [-1, 512, 14, 14]               0
           Conv2d-65          [-1, 512, 28, 28]       2,359,808
      BatchNorm2d-66          [-1, 512, 28, 28]           1,024
             ReLU-67          [-1, 512, 28, 28]               0
       ConvBnRelu-68          [-1, 512, 28, 28]               0
           Conv2d-69          [-1, 512, 28, 28]       2,359,808
      BatchNorm2d-70          [-1, 512, 28, 28]           1,024
             ReLU-71          [-1, 512, 28, 28]               0
       ConvBnRelu-72          [-1, 512, 28, 28]               0
           Conv2d-73          [-1, 256, 28, 28]       1,179,904
      BatchNorm2d-74          [-1, 256, 28, 28]             512
             ReLU-75          [-1, 256, 28, 28]               0
       ConvBnRelu-76          [-1, 256, 28, 28]               0
           Conv2d-77          [-1, 256, 56, 56]         590,080
      BatchNorm2d-78          [-1, 256, 56, 56]             512
             ReLU-79          [-1, 256, 56, 56]               0
       ConvBnRelu-80          [-1, 256, 56, 56]               0
           Conv2d-81          [-1, 256, 56, 56]         590,080
      BatchNorm2d-82          [-1, 256, 56, 56]             512
             ReLU-83          [-1, 256, 56, 56]               0
       ConvBnRelu-84          [-1, 256, 56, 56]               0
           Conv2d-85          [-1, 128, 56, 56]         295,040
      BatchNorm2d-86          [-1, 128, 56, 56]             256
             ReLU-87          [-1, 128, 56, 56]               0
       ConvBnRelu-88          [-1, 128, 56, 56]               0
           Conv2d-89        [-1, 128, 112, 112]         147,584
      BatchNorm2d-90        [-1, 128, 112, 112]             256
             ReLU-91        [-1, 128, 112, 112]               0
       ConvBnRelu-92        [-1, 128, 112, 112]               0
           Conv2d-93         [-1, 64, 112, 112]          73,792
      BatchNorm2d-94         [-1, 64, 112, 112]             128
             ReLU-95         [-1, 64, 112, 112]               0
       ConvBnRelu-96         [-1, 64, 112, 112]               0
           Conv2d-97         [-1, 64, 224, 224]          36,928
      BatchNorm2d-98         [-1, 64, 224, 224]             128
             ReLU-99         [-1, 64, 224, 224]               0
      ConvBnRelu-100         [-1, 64, 224, 224]               0
          Conv2d-101          [-1, 2, 224, 224]           1,154
          Conv2d-102          [-1, 5, 224, 224]           2,885
================================================================
Total params: 29,447,047
Trainable params: 29,447,047
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 688.68
Params size (MB): 112.33
Estimated Total Size (MB): 801.59
----------------------------------------------------------------
```

### References
https://github.com/nyoki-mtl/pytorch-discriminative-loss  
[Paper: Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/pdf/1708.02551.pdf)

