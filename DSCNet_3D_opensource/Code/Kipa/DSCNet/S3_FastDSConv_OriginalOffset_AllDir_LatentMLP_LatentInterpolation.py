   # -*- coding: utf-8 -*-
import torch
from torch import nn, cat


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

class EncoderMLP(nn.Module):
    def __init__(self, in_ch, k):
        super().__init__()
        self.net = nn.Sequential(
            # create "linear layer" over the flattened k*k*k*in_ch patch
            nn.Conv3d(in_ch, k, kernel_size=k, padding=k // 2, padding_mode="replicate"),
            nn.ReLU(),
            # MLP layers act on each voxel independently
            nn.Conv3d(k, k, kernel_size=1),
        )

    def forward(self, x):          # x: (N, C, D, W, H)
        return self.net(x)         # out: (N, 3K, D, W, H)

class DecoderMLP(nn.Module):
    def __init__(self, in_ch, k):
        super().__init__()
        self.net = nn.Sequential(
            # Squash input into k latent channels
            nn.Conv3d(in_ch, k, kernel_size=k, padding=k // 2, padding_mode="replicate"),
            nn.ReLU(),
        )

    def forward(self, x):          # x: (N, C, D, W, H)
        return self.net(x)         # out: (N, K, D, W, H)

class InterpolaterMLP(nn.Module):
    def __init__(self, in_ch, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(k+1, in_ch, kernel_size=1),
        )

    def forward(self, x):          # x: (N, K+1, D, W, H)
        return self.net(x)         # out: (N, C, D, W, H)

class DCN_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device):
        super(DCN_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.offset_enc_conv = EncoderMLP(in_ch, self.kernel_size) # takes in [N, C, D, W, H] and returns [N, K, D, W, H]
        self.offset_dec_conv = DecoderMLP(in_ch, self.kernel_size) # takes in [N, C, D, W, H] and returns [N, K, D, W, H]
        self.interp_convs = nn.ModuleList([InterpolaterMLP(in_ch, self.kernel_size) for _ in range(self.kernel_size)]) # takes in K sets of [N, K+1, D, W, H] and returns K sets of [N, C, D, W, H]
        self.device = device

        self.if_offset = if_offset
        self.morph = morph
        self.extend_scope = extend_scope

        self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=(self.kernel_size, 1, 1), stride=(self.kernel_size, 1, 1), padding=0)  # conv in z-direction; takes in [N, C_in, K*D, W, H], and returns [N, C_out, D, W, H]
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.dcn = None


    def forward(self, f):
        input_shape = f.shape # shape: [N, C, D, W, H];

        if self.dcn is None:
            self.dcn = DCN(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device, self.offset_dec_conv)

        # Encode
        l_offset = self.offset_enc_conv(f) # Output: [N, K, D, W, H]
        l_offset = torch.tanh(l_offset) # Output: [N, K, D, W, H]; tanh is (-1, 1)

        # constrain offsets to snake shape
        constrained_offset = self.dcn._constrain_offset(l_offset) # Output: [N, K, D, W, H]

        # Decode
        #decoded_offset = self.offset_dec_conv(constrained_offset) # Output: [N, 3*K, D, W, H]
        #decoded_offset = torch.clamp(decoded_offset, min=-self.kernel_size/2, max=self.kernel_size/2) # Output: [N, 3*K, D, W, H]

        # convert offsets to actual coordinates
        #z, y, x = self.dcn._coordinate_map_3D(decoded_offset) # Output: 3x [N, K, D, W, H]

        # retrieve output from coordinates via bilinear interpolation
        #deformed_feature = self.dcn._vectorized_new_bilinear_interpolate_3D(f, z, y, x) # Output: [N, C, K*D, W, H]

        latent_input = self.offset_dec_conv(f)
        deformed_feature = torch.cat([self.interp_convs[i](torch.cat([latent_input, constrained_offset[:, i, :, :, :].unsqueeze(1)], dim=1)) for i in range(self.kernel_size)], dim=2)

        out = self.dcn_conv(deformed_feature)
        out = self.gn(out)
        out = self.relu(out)
        return out

class DCN(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device, decoder_conv):
        self.num_points = kernel_size # K
        self.depth = input_shape[2] # D
        self.width = input_shape[3] # W
        self.height = input_shape[4] # H
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope
        self.num_channels = input_shape[1] # C
        self.decoder_conv = decoder_conv

    '''
    input: latent offset [N,K,D,W,H]
    output: constrained latent offset [N,K,D,W,H]
    '''
    def _constrain_offset(self, l_offset):
        c_offset = torch.zeros_like(l_offset).to(self.device)

        c_offset = c_offset.permute(1, 0, 2, 3, 4) # [K, N, D, W, H] permute to prepare for offset in kernel direction
        l_offset = l_offset.permute(1, 0, 2, 3, 4) # [K, N, D, W, H] 
        center = int(self.num_points // 2)

        c_offset[center + 1:] = torch.cumsum(l_offset[center + 1:], dim=0)
        c_offset[:center] = torch.flip(torch.cumsum(torch.flip(l_offset[:center], dims=[0]), dim=0), dims=[0])

        c_offset = c_offset.permute(1, 0, 2, 3, 4).to(self.device) # [N, K, D, W, H]

        return c_offset

    '''
    input: constrained decoded offset [N,3*K,D,W,H]
    output: 3x coordinate maps [N,K,D,W,H]
    '''
    def _coordinate_map_3D(self, d_offset):
        num_batch = d_offset.shape[0] # N

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

        z_new = z_center.to(self.device) # (N, K, D, W, H)
        y_new = y_center.to(self.device) # (N, K, D, W, H)
        x_new = x_center.to(self.device) # (N, K, D, W, H)

        z_offset, y_offset, x_offset = torch.split(d_offset, self.num_points, dim=1)

        z_new = z_new.add(z_offset) # add z_offset to z_new (which is all zeros except for depth)
        y_new = y_new.add(y_offset) # add y_offset to y_new (which is all zeros except for width)
        x_new = x_new.add(x_offset) # add y_offset to y_new (which is all zeros except for width)

        return z_new, y_new, x_new

    '''
    input: input feature map [N,C,D,W,H]；coordinate maps [N,K,D,W,H] 
    output: [N,C,D,W,K*H] or [N,C,D,K*W,H] or [N,C,K*D,W,H] deformed feature map
    '''
    def _vectorized_new_bilinear_interpolate_3D(self, input_feature, z, y, x):
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

        #outputs = outputs.reshape(N, C, K, D, W, H)

        #if self.morph == 0:
            #outputs = outputs.permute(0, 1, 3, 4, 2, 5)
            #outputs = outputs.reshape(N, C, D, W, K*H)
        #elif self.morph == 1:
            #outputs = outputs.permute(0, 1, 3, 2, 4, 5)
            #outputs = outputs.reshape(N, C, D, K*W, H)
        #else:
            #outputs = outputs.reshape(N, C, K*D, W, H)

        return outputs
