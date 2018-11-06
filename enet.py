# refer from : https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 bias=False):
        super(InitialBlock, self).__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=output_ch - 3,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(output_ch - 3),
            nn.PReLU()
        )
        self.ext_branch = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        out = torch.cat((main, ext), dim=1) # N, C, H, W

        return out


class RegularBottleNeck(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 projection_ratio=4,
                 regularizer_prob=0,
                 dilation=0,
                 assymmetric=False,
                 bias=False):
        super(RegularBottleNeck, self).__init__()

        reduced_depth = input_ch // projection_ratio

        self.ext_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=reduced_depth,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(reduced_depth),
            nn.PReLU()
        )

        if dilation:
            self.ext_branch_2 = nn.Sequential(
                nn.Conv2d(in_channels=reduced_depth,
                          out_channels=reduced_depth,
                          kernel_size=3,
                          stride=1,
                          padding=dilation,
                          bias=bias,
                          dilation=dilation),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU()
            )

        elif assymmetric:
            self.ext_branch_2 = nn.Sequential(
                nn.Conv2d(in_channels=reduced_depth,
                          out_channels=reduced_depth,
                          kernel_size=(5, 1),
                          stride=1,
                          padding=(2, 0),
                          bias=bias),
                nn.Conv2d(in_channels=reduced_depth,
                          out_channels=reduced_depth,
                          kernel_size=(1, 5),
                          stride=1,
                          padding=(0, 2),
                          bias=bias),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU()
            )

        else: # Regular Bottle Neck
            self.ext_branch_2 = nn.Sequential(
                nn.Conv2d(in_channels=reduced_depth,
                          out_channels=reduced_depth,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=bias),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU()
            )

        self.ext_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=reduced_depth,
                      out_channels=output_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(output_ch),
            nn.PReLU()
        )

        self.regularizer = nn.Dropout2d(p=regularizer_prob)

        self.prelu = nn.PReLU()

    def forward(self, x):
        main = x
        ext = self.ext_branch_1(x)
        ext = self.ext_branch_2(ext)
        ext = self.ext_branch_3(ext)
        ext = self.regularizer(ext)

        out = self.prelu(main + ext)

        return out


class DownSampleBottleNeck(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 projection_ratio=4,
                 regularizer_prob=0,
                 bias=False):
        super(DownSampleBottleNeck, self).__init__()

        reduced_depth = input_ch // projection_ratio

        self.ext_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=reduced_depth,
                      kernel_size=2,
                      stride=2,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(reduced_depth),
            nn.PReLU()
        )

        self.ext_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=reduced_depth,
                      out_channels=reduced_depth,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(reduced_depth),
            nn.PReLU()
        )

        self.ext_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=reduced_depth,
                      out_channels=output_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(output_ch),
            nn.PReLU()
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.regularizer = nn.Dropout2d(p=regularizer_prob)

        self.prelu = nn.PReLU()

    def forward(self, x):
        main, ind = self.max_pool(x)
        ext = self.ext_branch_1(x)
        ext = self.ext_branch_2(ext)
        ext = self.ext_branch_3(ext)
        ext = self.regularizer(ext)

        # Feature map padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.autograd.Variable(torch.zeros(n, ch_ext-ch_main, h, w))
        if main.is_cuda:
            padding = padding.cuda()
        main = torch.cat((main, padding), dim=1)

        out = self.prelu(main + ext)

        return out, ind


class UpSampleBottleNeck(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 projection_ratio=4,
                 regularizer_prob=0,
                 bias=False):
        super(UpSampleBottleNeck, self).__init__()

        reduced_depth = input_ch // projection_ratio

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=output_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(output_ch)
        )

        self.ext_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=reduced_depth,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(reduced_depth),
            nn.PReLU()
        )

        self.ext_branch_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=reduced_depth,
                      out_channels=reduced_depth,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      output_padding=1,
                      bias=bias),
            nn.BatchNorm2d(reduced_depth),
            nn.PReLU()
        )

        self.ext_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=reduced_depth,
                      out_channels=output_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(output_ch),
            nn.PReLU()
        )
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.regularizer = nn.Dropout2d(p=regularizer_prob)

        self.prelu = nn.PReLU()

    def forward(self, x, ind):
        main = self.main_branch(x)
        main = self.max_unpool(main, ind)
        ext = self.ext_branch_1(x)
        ext = self.ext_branch_2(ext)
        ext = self.ext_branch_3(ext)
        ext = self.regularizer(ext)

        out = self.prelu(main + ext)

        return out


class ENet(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(ENet, self).__init__()

        # Initial
        self.initial_block = InitialBlock(input_ch=input_ch, output_ch=16)

        # Shared Encoder
        # BottleNeck1
        self.bottleNeck1_0 = DownSampleBottleNeck(input_ch=16, output_ch=64, regularizer_prob=0.01)
        self.bottleNeck1_1 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.01)
        self.bottleNeck1_2 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.01)
        self.bottleNeck1_3 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.01)
        self.bottleNeck1_4 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.01)

        # BottleNeck2
        self.bottleNeck2_0 = DownSampleBottleNeck(input_ch=64, output_ch=128, regularizer_prob=0.1)
        self.bottleNeck2_1 = RegularBottleNeck(input_ch=128, output_ch=128, regularizer_prob=0.1)
        self.bottleNeck2_2 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=2, regularizer_prob=0.1)
        self.bottleNeck2_3 = RegularBottleNeck(input_ch=128, output_ch=128, assymmetric=True, regularizer_prob=0.1)
        self.bottleNeck2_4 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=4, regularizer_prob=0.1)
        self.bottleNeck2_5 = RegularBottleNeck(input_ch=128, output_ch=128, regularizer_prob=0.1)
        self.bottleNeck2_6 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=8, regularizer_prob=0.1)
        self.bottleNeck2_7 = RegularBottleNeck(input_ch=128, output_ch=128, assymmetric=True, regularizer_prob=0.1)
        self.bottleNeck2_8 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=16, regularizer_prob=0.1)

        # Binary Segmentation
        # BottleNeck3
        self.semBottleNeck3_0 = RegularBottleNeck(input_ch=128, output_ch=128, regularizer_prob=0.1)
        self.semBottleNeck3_1 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=2, regularizer_prob=0.1)
        self.semBottleNeck3_2 = RegularBottleNeck(input_ch=128, output_ch=128, assymmetric=True, regularizer_prob=0.1)
        self.semBottleNeck3_3 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=4, regularizer_prob=0.1)
        self.semBottleNeck3_4 = RegularBottleNeck(input_ch=128, output_ch=128, regularizer_prob=0.1)
        self.semBottleNeck3_5 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=8, regularizer_prob=0.1)
        self.semBottleNeck3_6 = RegularBottleNeck(input_ch=128, output_ch=128, assymmetric=True, regularizer_prob=0.1)
        self.semBottleNeck3_7 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=16, regularizer_prob=0.1)

        # BottleNeck4
        self.semBottleNeck4_0 = UpSampleBottleNeck(input_ch=128, output_ch=64, regularizer_prob=0.1)
        self.semBottleNeck4_1 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.1)
        self.semBottleNeck4_2 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.1)

        # BottleNeck5
        self.semBottleNeck5_0 = UpSampleBottleNeck(input_ch=64, output_ch=16, regularizer_prob=0.1)
        self.semBottleNeck5_1 = RegularBottleNeck(input_ch=16, output_ch=16, regularizer_prob=0.1)

        self.sem_out = nn.ConvTranspose2d(in_channels=16,
                                          out_channels=output_ch,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          bias=False)

        # Instance Segmentation
        # BottleNeck3
        self.insBottleNeck3_0 = RegularBottleNeck(input_ch=128, output_ch=128, regularizer_prob=0.1)
        self.insBottleNeck3_1 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=2, regularizer_prob=0.1)
        self.insBottleNeck3_2 = RegularBottleNeck(input_ch=128, output_ch=128, assymmetric=True, regularizer_prob=0.1)
        self.insBottleNeck3_3 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=4, regularizer_prob=0.1)
        self.insBottleNeck3_4 = RegularBottleNeck(input_ch=128, output_ch=128, regularizer_prob=0.1)
        self.insBottleNeck3_5 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=8, regularizer_prob=0.1)
        self.insBottleNeck3_6 = RegularBottleNeck(input_ch=128, output_ch=128, assymmetric=True, regularizer_prob=0.1)
        self.insBottleNeck3_7 = RegularBottleNeck(input_ch=128, output_ch=128, dilation=16, regularizer_prob=0.1)

        # BottleNeck4
        self.insBottleNeck4_0 = UpSampleBottleNeck(input_ch=128, output_ch=64, regularizer_prob=0.1)
        self.insBottleNeck4_1 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.1)
        self.insBottleNeck4_2 = RegularBottleNeck(input_ch=64, output_ch=64, regularizer_prob=0.1)

        # BottleNeck5
        self.insBottleNeck5_0 = UpSampleBottleNeck(input_ch=64, output_ch=16, regularizer_prob=0.1)
        self.insBottleNeck5_1 = RegularBottleNeck(input_ch=16, output_ch=16, regularizer_prob=0.1)

        self.ins_out = nn.ConvTranspose2d(in_channels=16,
                                          out_channels=5,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          bias=False)


    def forward(self, x):
        # Initial
        x = self.initial_block(x)

        # Shared Encoder
        # Stage1
        x, ind_1 = self.bottleNeck1_0(x)
        x = self.bottleNeck1_1(x)
        x = self.bottleNeck1_2(x)
        x = self.bottleNeck1_3(x)
        x = self.bottleNeck1_4(x)

        # Stage2
        x, ind_2 = self.bottleNeck2_0(x)
        x = self.bottleNeck2_1(x)
        x = self.bottleNeck2_2(x)
        x = self.bottleNeck2_3(x)
        x = self.bottleNeck2_4(x)
        x = self.bottleNeck2_5(x)
        x = self.bottleNeck2_6(x)
        x = self.bottleNeck2_7(x)
        x = self.bottleNeck2_8(x)

        # Binary Segmentation
        # Stage3
        x1 = self.semBottleNeck3_0(x)
        x1 = self.semBottleNeck3_1(x1)
        x1 = self.semBottleNeck3_2(x1)
        x1 = self.semBottleNeck3_3(x1)
        x1 = self.semBottleNeck3_4(x1)
        x1 = self.semBottleNeck3_5(x1)
        x1 = self.semBottleNeck3_6(x1)
        x1 = self.semBottleNeck3_7(x1)

        # Stage4
        x1 = self.semBottleNeck4_0(x1, ind_2)
        x1 = self.semBottleNeck4_1(x1)
        x1 = self.semBottleNeck4_2(x1)

        # Stage5
        x1 = self.semBottleNeck5_0(x1, ind_1)
        x1 = self.semBottleNeck5_1(x1)

        # Instance Segmentation
        # Stage3
        x2 = self.semBottleNeck3_0(x)
        x2 = self.semBottleNeck3_1(x2)
        x2 = self.semBottleNeck3_2(x2)
        x2 = self.semBottleNeck3_3(x2)
        x2 = self.semBottleNeck3_4(x2)
        x2 = self.semBottleNeck3_5(x2)
        x2 = self.semBottleNeck3_6(x2)
        x2 = self.semBottleNeck3_7(x2)

        # Stage4
        x2 = self.semBottleNeck4_0(x2, ind_2)
        x2 = self.semBottleNeck4_1(x2)
        x2 = self.semBottleNeck4_2(x2)

        # Stage5
        x2 = self.semBottleNeck5_0(x2, ind_1)
        x2 = self.semBottleNeck5_1(x2)

        # Stage 6
        sem = self.sem_out(x1)
        ins = self.ins_out(x2)

        return sem, ins
