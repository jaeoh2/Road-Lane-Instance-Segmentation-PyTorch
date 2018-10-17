import torch
import torch.nn as nn
import torch.nn.functional as F

class BinSegNet(nn.Module):
    # refer from : https://github.com/delta-onera/segnet_pytorch/blob/master/segnet.py
    def __init__(self, input_ch, output_ch):
        super(BinSegNet, self).__init__()
        # Encoder
        self.conv11 = nn.Conv2d(in_channels=input_ch, out_channels=64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64 ,64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)
        
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)
        
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)
        
        # Decoder
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512)
        
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)
        
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)
        
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64)
        
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64, output_ch, kernel_size=3, padding=1)
                
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x, ind_1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x, ind_2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x, ind_3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        x, ind_4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        x, ind_5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder
        x = F.max_unpool2d(x, ind_5, kernel_size=2, stride=2)
        x = F.relu(self.bn53d(self.conv53d(x)))
        x = F.relu(self.bn52d(self.conv52d(x)))
        x = F.relu(self.bn51d(self.conv51d(x)))
        
        x = F.max_unpool2d(x, ind_4, kernel_size=2, stride=2)
        x = F.relu(self.bn43d(self.conv43d(x)))
        x = F.relu(self.bn42d(self.conv42d(x)))
        x = F.relu(self.bn41d(self.conv41d(x)))
        
        x = F.max_unpool2d(x, ind_3, kernel_size=2, stride=2)
        x = F.relu(self.bn33d(self.conv33d(x)))
        x = F.relu(self.bn32d(self.conv32d(x)))
        x = F.relu(self.bn31d(self.conv31d(x)))
        
        x = F.max_unpool2d(x, ind_2, kernel_size=2, stride=2)
        x = F.relu(self.bn22d(self.conv22d(x)))
        x = F.relu(self.bn21d(self.conv21d(x)))
        
        x = F.max_unpool2d(x, ind_1, kernel_size=2, stride=2)
        x = F.relu(self.bn12d(self.conv12d(x)))
        x = self.conv11d(x)
        x = F.softmax(x, dim=1)
        
        return x

