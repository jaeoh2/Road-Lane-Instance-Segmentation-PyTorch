import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=3, padding=1):
        super(ConvBnRelu, self).__init__()
        self.conv =  nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SegNet(nn.Module):
    # refer from : https://github.com/delta-onera/segnet_pytorch/blob/master/segnet.py
    def __init__(self, input_ch, output_ch):
        super(SegNet, self).__init__()
        #Encoder
        self.enc11 = ConvBnRelu(input_ch, 64)
        self.enc12 = ConvBnRelu(64, 64)

        self.enc21 = ConvBnRelu(64, 128)
        self.enc22 = ConvBnRelu(128, 128)

        self.enc31 = ConvBnRelu(128, 256)
        self.enc32 = ConvBnRelu(256, 256)
        self.enc33 = ConvBnRelu(256, 256)

        self.enc41 = ConvBnRelu(256, 512)
        self.enc42 = ConvBnRelu(512, 512)
        self.enc43 = ConvBnRelu(512, 512)

        self.enc51 = ConvBnRelu(512, 512)
        self.enc52 = ConvBnRelu(512, 512)
        self.enc53 = ConvBnRelu(512, 512)

        #Decodr
        self.dec53 = ConvBnRelu(512, 512)
        self.dec52 = ConvBnRelu(512, 512)
        self.dec51 = ConvBnRelu(512, 512)

        self.dec43 = ConvBnRelu(512, 512)
        self.dec42 = ConvBnRelu(512, 512)
        self.dec41 = ConvBnRelu(512, 256)

        self.dec33 = ConvBnRelu(256, 256)
        self.dec32 = ConvBnRelu(256, 256)
        self.dec31 = ConvBnRelu(256, 128)

        self.dec22 = ConvBnRelu(128, 128)
        self.dec21 = ConvBnRelu(128, 64)

        self.dec12 = ConvBnRelu(64, 64)

        self.sem_out = nn.Conv2d(64, output_ch, kernel_size=3, stride=1, padding=1)
        self.ins_out = nn.Conv2d(64, 5, kernel_size=3, stride=1, padding=1)
                
    def forward(self, x):
        #Encoder
        x = self.enc11(x)
        x = self.enc12(x)
        x, ind_1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.enc21(x)
        x = self.enc22(x)
        x, ind_2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.enc31(x)
        x = self.enc32(x)
        x = self.enc33(x)
        x, ind_3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.enc41(x)
        x = self.enc42(x)
        x = self.enc43(x)
        x, ind_4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.enc51(x)
        x = self.enc52(x)
        x = self.enc53(x)
        x, ind_5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        #Decoder
        x = F.max_unpool2d(x, ind_5, kernel_size=2, stride=2)
        x = self.dec53(x)
        x = self.dec52(x)
        x = self.dec51(x)

        x = F.max_unpool2d(x, ind_4, kernel_size=2, stride=2)
        x = self.dec43(x)
        x = self.dec42(x)
        x = self.dec41(x)

        x = F.max_unpool2d(x, ind_3, kernel_size=2, stride=2)
        x = self.dec33(x)
        x = self.dec32(x)
        x = self.dec31(x)

        x = F.max_unpool2d(x, ind_2, kernel_size=2, stride=2)
        x = self.dec22(x)
        x = self.dec21(x)

        x = F.max_unpool2d(x, ind_1, kernel_size=2, stride=2)
        x = self.dec12(x)

        sem = F.softmax(self.sem_out(x), dim=1)
        ins = F.softmax(self.ins_out(x), dim=1)

        return sem, ins 

