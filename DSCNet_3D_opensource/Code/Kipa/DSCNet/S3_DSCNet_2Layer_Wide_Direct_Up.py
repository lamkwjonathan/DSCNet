# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
from S3_FastDSConv_OriginalOffset_AllDir_LatentMLP_LatentInterpolation import DCN_Conv


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__() 

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class ScheduledDropout(nn.Module):
    def __init__(self, p_start=0.5, p_end=0.0, p_decay_power=0.9, total_steps=100):
        super().__init__()
        self.p_start       = p_start
        self.p_end         = p_end
        self.p_decay_power = p_decay_power
        self.total_steps   = total_steps
        self.step          = 1

    def get_p(self):
        fraction = min(self.step / self.total_steps, 1.0)
        return self.p_end + (self.p_start - self.p_end) * (1 - fraction) ** self.p_decay_power # polynomial decay

    def forward(self, x):
        p = self.get_p()
        self.step += 1

        if not self.training or p == 0.0:
            return x

        return nn.functional.dropout(x, p=p, training=True)

class DSCNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernel_size, extend_scope, if_offset, device, number, dim, epochs):
        super(DSCNet, self).__init__()
        self.device = device
        self.kernel_size = kernel_size # kernel_size not in use
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        self.dim = dim
        self.epochs = epochs

        # Unet
        self.conv00 = EncoderConv(n_channels, 2*self.number)
        self.conv0a = DCN_Conv(n_channels, self.number, 9, self.extend_scope, 0, self.if_offset, self.device)
        self.conv0aa = DCN_Conv(n_channels, self.number, 9, self.extend_scope, 0, self.if_offset, self.device)
        self.conv0b = DCN_Conv(n_channels, self.number, 5, self.extend_scope, 1, self.if_offset, self.device)
        self.conv0bb = DCN_Conv(n_channels, self.number, 5, self.extend_scope, 1, self.if_offset, self.device)
        self.conv0c = DCN_Conv(n_channels, self.number, 7, self.extend_scope, 2, self.if_offset, self.device)
        self.conv0cc = DCN_Conv(n_channels, self.number, 7, self.extend_scope, 2, self.if_offset, self.device)
        self.conv1 = EncoderConv(8*self.number, self.number)

        self.conv20 = EncoderConv(n_channels+self.number, 2*self.number)
        self.conv2a = DCN_Conv(n_channels+self.number, 2*self.number, 9, self.extend_scope, 0, self.if_offset, self.device)
        self.conv2b = DCN_Conv(n_channels+self.number, 2*self.number, 5, self.extend_scope, 1, self.if_offset, self.device)
        self.conv2c = DCN_Conv(n_channels+self.number, 2*self.number, 7, self.extend_scope, 2, self.if_offset, self.device)
        self.conv3 = EncoderConv(8*self.number, 2*self.number)

        self.conv40 = EncoderConv(n_channels+2*self.number, 4*self.number)
        self.conv4a = DCN_Conv(n_channels+2*self.number, 4*self.number, 9, self.extend_scope, 0, self.if_offset, self.device)
        self.conv4b = DCN_Conv(n_channels+2*self.number, 4*self.number, 5, self.extend_scope, 1, self.if_offset, self.device)
        self.conv4c = DCN_Conv(n_channels+2*self.number, 4*self.number, 7, self.extend_scope, 2, self.if_offset, self.device)
        self.conv5 = EncoderConv(16*self.number, 4*self.number)

        self.conv140 = DecoderConv(n_channels+6*self.number, 2*self.number)
        self.conv14a = DCN_Conv(n_channels+6*self.number, 2*self.number, 9, self.extend_scope, 0, self.if_offset, self.device)
        self.conv14b = DCN_Conv(n_channels+6*self.number, 2*self.number, 5, self.extend_scope, 1, self.if_offset, self.device)
        self.conv14c = DCN_Conv(n_channels+6*self.number, 2*self.number, 7, self.extend_scope, 2, self.if_offset, self.device)
        self.conv15 = DecoderConv(8*self.number, 2*self.number)

        self.conv160 = DecoderConv(n_channels+3*self.number, self.number)
        self.conv16a = DCN_Conv(n_channels+3*self.number, self.number, 9, self.extend_scope, 0, self.if_offset, self.device)
        self.conv16b = DCN_Conv(n_channels+3*self.number, self.number, 5, self.extend_scope, 1, self.if_offset, self.device)
        self.conv16c = DCN_Conv(n_channels+3*self.number, self.number, 7, self.extend_scope, 2, self.if_offset, self.device)
        self.conv17 = DecoderConv(4*self.number, self.number)

        self.out_conv = nn.Conv3d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        #self.dropout = nn.Dropout(0.5) # Original: 0.5
        #self.dropout = ScheduledDropout(total_steps=epochs)

    def forward(self, x):
        # block0
        # x = self.maxpooling(x)
        x_00_0 = self.conv00(x)
        x_0a_0 = self.conv0a(x)
        x_0a_1 = self.conv0aa(x)
        x_0b_0 = self.conv0b(x)
        x_0b_1 = self.conv0bb(x)
        x_0c_0 = self.conv0c(x)
        x_0c_1 = self.conv0cc(x)
        x_0_1 = self.conv1(cat([x_00_0, x_0a_0, x_0a_1, x_0b_0, x_0b_1, x_0c_0, x_0c_1], dim=1))
        #x_0_1 = self.dropout(x_0_1)

        # block1
        x_a = self.maxpooling(x)
        x_0_1a = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(cat([x_a, x_0_1a], dim=1))
        x_2a_0 = self.conv2a(cat([x_a, x_0_1a], dim=1))
        x_2b_0 = self.conv2b(cat([x_a, x_0_1a], dim=1))
        x_2c_0 = self.conv2c(cat([x_a, x_0_1a], dim=1))
        x_1_1 = self.conv3(cat([x_20_0, x_2a_0, x_2b_0, x_2c_0], dim=1))
        #x_1_1 = self.dropout(x_1_1)

        # block2
        x_b = self.maxpooling(x_a)
        x_1_1a = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(cat([x_b, x_1_1a], dim=1))
        x_4a_0 = self.conv4a(cat([x_b, x_1_1a], dim=1))
        x_4b_0 = self.conv4b(cat([x_b, x_1_1a], dim=1))
        x_4c_0 = self.conv4c(cat([x_b, x_1_1a], dim=1))
        x_2_1 = self.conv5(cat([x_40_0, x_4a_0, x_4b_0, x_4c_0], dim=1))
        #x_2_1 = self.dropout(x_2_1)

        # block5
        x_2_1a = self.up(x_2_1)
        x_140_2 = self.conv140(cat([x_a, x_2_1a, x_1_1], dim=1))
        x_14a_2 = self.conv14a(cat([x_a, x_2_1a, x_1_1], dim=1))
        x_14b_2 = self.conv14b(cat([x_a, x_2_1a, x_1_1], dim=1))
        x_14c_2 = self.conv14c(cat([x_a, x_2_1a, x_1_1], dim=1))
        x_1_3 = self.conv15(cat([x_140_2, x_14a_2, x_14b_2, x_14c_2], dim=1))
        #x_1_3 = self.dropout(x_1_3)

        # block6
        x_1_3a = self.up(x_1_3)
        x_160_2 = self.conv160(cat([x, x_1_3a, x_0_1], dim=1))
        x_16a_2 = self.conv16a(cat([x, x_1_3a, x_0_1], dim=1))
        x_16b_2 = self.conv16b(cat([x, x_1_3a, x_0_1], dim=1))
        x_16c_2 = self.conv16c(cat([x, x_1_3a, x_0_1], dim=1))
        x_0_3 = self.conv17(cat([x_160_2, x_16a_2, x_16b_2, x_16c_2], dim=1))
        #x_0_3 = self.dropout(x_0_3)
        out = self.out_conv(x_0_3)
        out = self.softmax(out)
        # out = self.up(out) # try this for upsampling

        return out
