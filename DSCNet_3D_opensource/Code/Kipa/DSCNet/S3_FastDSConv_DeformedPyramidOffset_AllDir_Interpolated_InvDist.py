   # -*- coding: utf-8 -*-
import torch
from torch import nn, cat
import numpy as np


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch) # Separates out_ch number of channels into out_ch // 4 number of groups (Each group has 4 channels), normalizes within group
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DCN_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device):
        super(DCN_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_input_list = np.abs(np.arange(self.kernel_size) - self.kernel_size // 2) 
        self.kernel_multiplier = torch.from_numpy(np.repeat(self.kernel_input_list, 3)).view(1, -1, 1, 1, 1).to(device)
        #self.offset_conv = nn.Conv3d(in_ch, 3 * self.kernel_size, self.kernel_size, padding=self.kernel_size//2) # takes in [N, C, D, W, H] and returns [N, 3*K, D, W, H]
        self.offset_convs = nn.ModuleList([nn.Conv3d(in_ch, 3, 1+i*2, padding=i) for i in self.kernel_input_list])
        self.bn = nn.BatchNorm3d(3 * self.kernel_size) # Normalizes across the channel dimension;
        self.device = device

        self.if_offset = if_offset
        self.morph = morph
        self.extend_scope = extend_scope

        #self.dcn_conv_x = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, self.kernel_size), stride=(1, 1, self.kernel_size), padding=0)  # conv in x-direction; takes in [N, C_in, D, W, K*H], and returns [N, C_out, D, W, H]
        #self.dcn_conv_y = nn.Conv3d(in_ch, out_ch, kernel_size=(1, self.kernel_size, 1), stride=(1, self.kernel_size, 1), padding=0)  # conv in y-direction; takes in [N, C_in, D, K*W, H], and returns [N, C_out, D, W, H]
        #self.dcn_conv_z = nn.Conv3d(in_ch, out_ch, kernel_size=(self.kernel_size, 1, 1), stride=(self.kernel_size, 1, 1), padding=0)  # conv in z-direction; takes in [N, C_in, K*D, W, H], and returns [N, C_out, D, W, H]

        #self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)
        self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=(2*self.kernel_size-1, 1, 1), stride=(2*self.kernel_size-1, 1, 1), padding=0)  # conv in z-direction; takes in [N, C_in, K*D, W, H], and returns [N, C_out, D, W, H]
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.dcn = None


    def forward(self, f):
        # Input: [N, K, D, W, H];
        #offset = self.offset_conv(f) # Output: [N, 3*K, D, W, H];
        offset = torch.cat([module(f) for module in self.offset_convs], dim=1) # Output: [N, 3*K, D, W, H];
        offset = self.bn(offset) # Output: [N, 3*K, D, W, H];
        #offset = torch.tanh(offset) * (self.kernel_size // 2) # Output: [N, 3*K, D, W, H]; tanh is (-1, 1)
        offset = torch.tanh(offset) * self.kernel_multiplier
        
        input_shape = f.shape # shape: [N, C, D, W, H];

        if self.dcn is None:
            self.dcn = DCN(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device, self.relu)

        deformed_feature = self.dcn.deform_conv(f, offset, self.if_offset) # _coordinate_map_3D (Output: [N, K, D, W, H]) + _vectorized_new_bilinear_interpolate_3D (Output: [N, C, K, D, W, H])

        # Only ever does one of the following
        # if self.morph == 0:
        #     x = self.dcn_conv_x(deformed_feature) # [N, C_out, D, W, H]
        #     x = self.gn(x)
        #     x = self.relu(x)
        #     return x
        # elif self.morph == 1:
        #     x = self.dcn_conv_y(deformed_feature) # [N, C_out, D, W, H]
        #     x = self.gn(x)
        #     x = self.relu(x)
        #     return x
        # else:
        #     x = self.dcn_conv_z(deformed_feature) # [N, C_out, D, W, H]
        #     x = self.gn(x)
        #     x = self.relu(x)  
        #     return x

        x = self.dcn_conv(deformed_feature)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DCN(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device, relu):
        self.num_points = kernel_size # K
        self.depth = input_shape[2] # D
        self.width = input_shape[3] # W
        self.height = input_shape[4] # H
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope
        self.num_channels = input_shape[1] # C
        self.relu = relu

    '''
    input: offset [N,2*K,D,W,H]
    output: [N,K,D,W,H]   coordinate map
    '''
    def _coordinate_map_3D(self, offset, if_offset):
        # variable num_batch
        num_batch = offset.shape[0] # N

        # offset
        #offset1, offset2, offset3 = torch.split(offset, self.num_points, dim=1) # Split offset1 into groups of self.num_points i.e. [N, K, D, W, H]
        offset1 = offset[:, 0::3, :, :, :] # [N, K, D, W, H]
        offset2 = offset[:, 1::3, :, :, :] # [N, K, D, W, H]
        offset3 = offset[:, 2::3, :, :, :] # [N, K, D, W, H]

        z_coords = torch.arange(self.depth) # [D]
        y_coords = torch.arange(self.width) # [W]
        x_coords = torch.arange(self.height) # [H]
        z_center, y_center, x_center = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij') # [D, W, H] where z increases in depth direction, y in width, and x in height.
        z_center = z_center.view(1, 1, self.depth, self.width, self.height) # [1, 1, D, W, H]
        y_center = y_center.view(1, 1, self.depth, self.width, self.height) # [1, 1, D, W, H]
        x_center = x_center.view(1, 1, self.depth, self.width, self.height) # [1, 1, D, W, H]
        z_center = z_center.expand(num_batch, self.num_points, self.depth, self.width, self.height).float() # [N, K, D, W, H]
        y_center = y_center.expand(num_batch, self.num_points, self.depth, self.width, self.height).float() # [N, K, D, W, H]
        x_center = x_center.expand(num_batch, self.num_points, self.depth, self.width, self.height).float() # [N, K, D, W, H]

        #x_spread = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points) # [K] running from -self.num_points//2 to self.num_points//2
        #x_spread = x_spread.view(1, self.num_points, 1, 1, 1) # [1, K, 1, 1, 1]
        #x_grid = x_spread.expand(num_batch, self.num_points, self.depth, self.width, self.height) # (N, K, D, W, H)

        z_new = z_center.to(self.device) # (N, K, D, W, H)
        y_new = y_center.to(self.device) # (N, K, D, W, H)
        x_new = x_center.to(self.device) # (N, K, D, W, H)

        z_offset = offset1.clone().to(self.device) # detach ensures that gradients do not propagate from z_offset to offset1; clone creates an independent matrix in memory
        y_offset = offset2.clone().to(self.device) # detach ensures that gradients do not propagate from y_offset to offset2; clone creates an independent matrix in memory
        x_offset = offset3.clone().to(self.device) # detach ensures that gradients do not propagate from y_offset to offset3; clone creates an independent matrix in memory
        
        #z_offset = torch.zeros_like(offset1).to(self.device)
        #y_offset = torch.zeros_like(offset2).to(self.device)
        #x_offset = torch.zeros_like(offset3).to(self.device)

        if if_offset:
            #z_offset = z_offset.permute(1, 0, 2, 3, 4) # [K, N, D, W, H] permute to prepare for offset in kernel direction
            #y_offset = y_offset.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
            #x_offset = x_offset.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
            #offset1 = offset1.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
            #offset2 = offset2.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
            #offset3 = offset3.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
            center = int(self.num_points // 2)
            z_offset[:, center, :, :, :] = 0
            y_offset[:, center, :, :, :] = 0
            x_offset[:, center, :, :, :] = 0
            #for index in range(1, center + 1):
                #z_offset[center + index] = z_offset[center + index - 1] + offset1[center + index] 
                #z_offset[center - index] = z_offset[center - index + 1] + offset1[center - index]
                #y_offset[center + index] = y_offset[center + index - 1] + offset2[center + index]
                #y_offset[center - index] = y_offset[center - index + 1] + offset2[center - index]
                #x_offset[center + index] = x_offset[center + index - 1] + offset3[center + index]
                #x_offset[center - index] = x_offset[center - index + 1] + offset3[center - index]

            #z_offset = z_offset.permute(1, 0, 2, 3, 4).to(self.device) # [N, K, D, W, H]
            #y_offset = y_offset.permute(1, 0, 2, 3, 4).to(self.device) # [N, K, D, W, H]
            #x_offset = x_offset.permute(1, 0, 2, 3, 4).to(self.device) # [N, K, D, W, H]
            z_new = z_new.add(z_offset) # add z_offset to z_new 
            y_new = y_new.add(y_offset) # add y_offset to y_new
            x_new = x_new.add(x_offset) # add y_offset to y_new

        #z_diff = torch.abs(z_new[:, 1:, :, :, :] - z_new[:, :-1, :, :, :])
        #z_diff = self.relu(z_diff - 1)
        #z_dist = torch.linalg.vector_norm(z_diff, dim=1).mean() / (self.num_points - 1)
        #y_diff = torch.abs(y_new[:, 1:, :, :, :] - y_new[:, :-1, :, :, :])
        #y_diff = self.relu(y_diff - 1)
        #y_dist = torch.linalg.vector_norm(y_diff, dim=1).mean() / (self.num_points - 1)
        #x_diff = torch.abs(x_new[:, 1:, :, :, :] - x_new[:, :-1, :, :, :])
        #x_diff = self.relu(x_diff - 1)
        #x_dist = torch.linalg.vector_norm(x_diff, dim=1).mean() / (self.num_points - 1)
        #dist_loss = (z_dist + y_dist + x_dist) / 3

        return z_new, y_new, x_new

    '''
    input: coordinate maps [N,K,D,W,H] 
    output: [N,2*K-1,D,W,H] interpolated coordinate maps
    '''
    def _coordinate_interpolate(self, z, y, x):
        N, K, D, W, H = z.shape

        # averages of consecutive channel pairs: shape [N, K-1, D, W, H]
        z_avg = (z[:, :-1, ...] + z[:, 1:, ...]) / 2
        y_avg = (y[:, :-1, ...] + y[:, 1:, ...]) / 2
        x_avg = (x[:, :-1, ...] + x[:, 1:, ...]) / 2

        z_dist = (z[:, :-1, ...] - z[:, 1:, ...])
        y_dist = (y[:, :-1, ...] - y[:, 1:, ...])
        x_dist = (x[:, :-1, ...] - x[:, 1:, ...])
        dist = torch.sqrt(torch.square(z_dist) + torch.square(y_dist) + torch.square(x_dist))
        dist = torch.tanh(dist)
        inv_dist = 1 - dist


        # interleave t[:, :-1] with avg, then append the final original channel
        # stacked shape: [N, K-1, 2, D, W, H] -> reshape -> [N, 2*(K-1), D, W, H]
        z_interleaved = torch.stack([z[:, :-1, ...], z_avg], dim=2).reshape(N, 2 * (K - 1), D, W, H)
        y_interleaved = torch.stack([y[:, :-1, ...], y_avg], dim=2).reshape(N, 2 * (K - 1), D, W, H)
        x_interleaved = torch.stack([x[:, :-1, ...], x_avg], dim=2).reshape(N, 2 * (K - 1), D, W, H)

        # append the last original channel (index K-1) at the end
        z_out = torch.cat([z_interleaved, z[:, -1:, ...]], dim=1)
        y_out = torch.cat([y_interleaved, y[:, -1:, ...]], dim=1)
        x_out = torch.cat([x_interleaved, x[:, -1:, ...]], dim=1)

        return z_out, y_out, x_out, inv_dist

    '''
    input: input feature map [N,C,D,W,H]；coordinate maps [N,K,D,W,H] 
    output: [N,C,K*D,W,H] deformed feature map
    '''
    def _vectorized_new_bilinear_interpolate_3D(self, input_feature, z, y, x, inv_dist):
        N, K, D, W, H = z.shape
        C = self.num_channels

        # Fold K in to D dimension
        input_feature = input_feature.unsqueeze(2) # [N, C, 1, D, W, H]
        input_feature = input_feature.expand(N, C, K, D, W, H) # [N, C, K, D, W, H]
        input_feature = input_feature.reshape(N, C, K*D, W, H).float() # [N, C, K*D, W, H]

        # Normalise coordinates to [-1, 1]
        z_norm = 2.0 * z / (D - 1) - 1.0
        y_norm = 2.0 * y / (W - 1) - 1.0
        x_norm = 2.0 * x / (H - 1) - 1.0
        
        # Build grid
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1) # [N, K, D, W, H, 3]
        grid = grid.reshape(N, K*D, W, H, 3).float() # [N, K*D, W, H, 3]

        outputs = torch.nn.functional.grid_sample(
            input_feature, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ) # [N, C, K*D, W, H]

        outputs = outputs.reshape(N, C, K, D, W, H)

        #if self.morph == 0:
            #outputs = outputs.permute(0, 1, 3, 4, 2, 5)
            #outputs = outputs.reshape(N, C, D, W, K*H)
        #elif self.morph == 1:
            #outputs = outputs.permute(0, 1, 3, 2, 4, 5)
            #outputs = outputs.reshape(N, C, D, K*W, H)
        #else:
            #outputs = outputs.reshape(N, C, K*D, W, H)

        #k_idx = torch.arange(K)
        #scale_per_k = torch.where(k_idx % 2 == 0, 0.5, 1.0)
        #scale = scale_per_k.repeat_interleave(D).view(1, 1, K * D, 1, 1).to(self.device)
        #outputs_scaled = outputs * scale

        even_part = outputs[:, :, 0::2, :, :, :]        # unchanged
        odd_part = outputs[:, :, 1::2, :, :, :] * inv_dist.view(N, 1, -1, D, W, H)
        outputs_scaled = torch.empty_like(outputs)
        outputs_scaled[:, :, 0::2, :, :, :] = even_part
        outputs_scaled[:, :, 1::2, :, :, :] = odd_part
        outputs_scaled = outputs_scaled.reshape(N, C, K * D, W, H)

        return outputs_scaled

    
    def deform_conv(self, input, offset, if_offset):
        z, y, x = self._coordinate_map_3D(offset, if_offset)
        z_ip, y_ip, x_ip, inv_dist = self._coordinate_interpolate(z, y, x)
        deformed_feature = self._vectorized_new_bilinear_interpolate_3D(input, z_ip, y_ip, x_ip, inv_dist)
        return deformed_feature
