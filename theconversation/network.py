import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
import torchvision
import pdb
import resnet


# model
class VideoNet(nn.Module):    
    def __init__(self):
        super(VideoNet, self).__init__()
        #videonet
        self.Conv3d = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), dilation=1, groups=1, bias=True)
        #time, width, height
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), dilation=1, return_indices=False, ceil_mode=False)
        self.resnet18 = resnet.resnet18(sample_size=32, sample_duration=1)
        
    def forward(self, video):
        features = self.Conv3d(video)
        features = self.bn1(features)
        features = self.relu(features)
        #features = self.maxpool(features)
        features = self.resnet18(features)
        return features

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, transpose=False):
        super(ResidualBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=True).cuda()
        self.bn1 = nn.BatchNorm1d(80).cuda()
        
        if not transpose:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, 
                         stride=stride, padding=2, bias=False).cuda()
        else:
            self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, 
                         stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
    def forward(self, x):
        residual = x.cuda()
        out = self.relu1(x)
        out = self.bn1(out)
        out = self.conv1(out)
        out += residual
        return out



# model
class MagnitudeSubNet(nn.Module):    
    def __init__(self):
        super(MagnitudeSubNet, self).__init__()

        self.LinearizeShape = nn.Linear(2048, 1600)
        self.outputLineaize = nn.Linear(40,20)

        self.vres = []
        #convolution along time dimension
        for i in range(10):
            if(i == 0):
                self.vres.append(ResidualBlock(80, 80))
            else:
                self.vres.append(ResidualBlock(80, 80))

        self.ares = []
        #audio
        #determine input channel dimension
        for i in range(5):
            if(i % 2 != 0):
                 self.ares.append(ResidualBlock(80, 80))
            else:
                self.ares.append(ResidualBlock(80, 80))

        
        self.magn_res = []
        for i in range(15):
            if((i + 1)% 5 == 0):
                self.magn_res.append(ResidualBlock(80, 80))
            else:
                self.magn_res.append(ResidualBlock(80, 80))

        
    def forward(self, visual, audio, batch_size):

        originalAudio = audio
        visual = self.LinearizeShape(visual)
        visual = visual.reshape((batch_size,80,20))

        for block in self.vres:
            visual = block(visual)

        for block in self.ares:
            audio = block(audio)

        # concatenation over channels
        magn_feats = torch.cat((visual, audio), dim = -1)

        for block in self.magn_res:
            magn_feats = block(magn_feats)

        #sigmoid over which dimension
        magn_feats = F.sigmoid(magn_feats)
        magn_feats = self.outputLineaize(magn_feats)

        return originalAudio * magn_feats

# model
class PhaseSubNet(nn.Module):    
    def __init__(self):
        super(PhaseSubNet, self).__init__()
        
        self.conv = []
        #convolution along time dimension
        for i in range(6):
            self.conv.append(ResidualBlock(80, 80))

        #dimension input = concat of conv
        #dimension output = same as noisy phase
        self.project = nn.Linear(40,20)
        self.eps = 0.00000000001
            

        
    def forward(self, noisy_phase, clean_magn):
        
        #check dimension of cat, channel
        fused_feats = torch.cat((clean_magn, noisy_phase), dim = -1)
        for block in self.conv:
            fused_feats = block(fused_feats)


        phase_residual = self.project(fused_feats)
        clean_phase = phase_residual + noisy_phase

        #check over dimension
        norm = clean_phase.norm(p=2, dim=1, keepdim=True)
        clean_phase = clean_phase.div(norm.expand_as(clean_phase))

        return clean_phase



