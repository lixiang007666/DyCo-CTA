import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch import SwinUnet3d


"""## EcoTTA Implementation for SwinUnet3d """

class simplify_SwinUnet3d(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc1 = model.enc[0]
        self.enc2 = model.enc[1]
        self.enc3 = model.enc[2]
        self.enc4 = model.enc[3]
        self.dec3 = model.dec[0]
        self.dec2 = model.dec[1]
        self.dec1 = model.dec[2]

        self.module_list = [self.enc2, self.enc3, self.enc4, \
                            self.dec3, self.dec2, self.dec1]
        self.classifier = model.classifier

    def forward(self, x):
        skip_connect = []

        out_enc1 = self.enc1(x)
        skip_connect.append(out_enc1)
        out_enc2 = self.enc2(out_enc1)
        skip_connect.append(out_enc2)
        out_enc3 = self.enc3(out_enc2)
        skip_connect.append(out_enc3)
        out = self.enc4(out_enc3)

        out = self.dec3(out, skip_connect.pop())
        out = self.dec2(out, skip_connect.pop())
        out = self.dec1(out, skip_connect.pop())

        out = self.classifier(out)
        return out

class conv_block3D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1):
        super().__init__()
        stride = int(1.0 / stride)
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, bias=False)
        # if stride < 1:
        #     stride = int(1.0 / stride)
        #     self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, bias=False)
        # else:
        #     assert (out_dim % 2) == 0
        #     self.conv1 = nn.Conv3d(out_dim, out_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        #     self.conv2 = nn.ConvTranspose3d(in_dim, out_dim // 2, kernel_size=kernel_size, stride=stride, bias=False)
        
        self.bn = nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class build_meta_block3D(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_scale=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scale = spatial_scale
        
        self.meta_bn = nn.BatchNorm3d(out_channels)
        self.conv_block = conv_block3D(in_channels, out_channels, kernel_size=2, stride=spatial_scale)
    
    def forward(self, x):
        out = self.conv_block(x)
        return out


class one_part_of_networks3D(nn.Module):
    def __init__(self, original_part, meta_part):
        super().__init__()
        self.original_part = original_part
        self.meta_part = meta_part
        self.btsloss = None
        self.cal_mseloss = False

    def forward(self, x):
        if not self.cal_mseloss:
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
        else:
            x = x.detach()
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
            loss = nn.L1Loss(reduction='none')
            self.btsloss = loss(out, out1.detach()).mean()
        return out


def attach_meta_networks_resunet3D(simplified_model, K=3):
    if K == 6:
        num_blocks = [1, 1, 1, 1, 1, 1]
    elif K == 3:
        num_blocks = [1, 1, 1]
    else:
        raise ValueError("K should be 6")
    
    if K == 6:
        in_channels_per_partition = [64, 256, 512, 1024, 512, 256]
        out_channels_per_partition = [256, 512, 1024, 512, 256, 64]
        spatial_scales_per_partition = [0.5, 0.5, 0.5, 2, 2, 2] 
    elif K == 3:
        in_channels_per_partition = [64, 256, 512]
        out_channels_per_partition = [256, 512, 1024]
        spatial_scales_per_partition = [0.5, 0.5, 0.5]         

    
    class ecotta_swinunet3d(nn.Module):
        def __init__(self, simplified_model, num_blocks, in_channels_per_partition, out_channels_per_partition, spatial_scales_per_partition):
            super().__init__()
            
            self.conv1 = simplified_model.enc1
            self.module_list = simplified_model.module_list
            self.classifier = simplified_model.classifier
            
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.meta_parts = nn.ModuleList()
            
            start_idx = 0
            for i, num_block in enumerate(num_blocks):
                modules_in_partition = self.module_list[start_idx:start_idx+num_block]

                meta_part = build_meta_block3D(
                    in_channels_per_partition[i], 
                    out_channels_per_partition[i], 
                    spatial_scales_per_partition[i]
                )

                original_part = modules_in_partition[0]
                wrapped_part = one_part_of_networks3D(original_part, meta_part)

                self.encoders.append(wrapped_part)
                self.meta_parts.append(meta_part)
                
                start_idx += num_block
            
            for decoder in self.module_list[start_idx:]:
                self.decoders.append(decoder)
                
        
        def forward(self, x):
            skip_connect = []

            out = self.conv1(x)
            skip_connect.append(out)
            out = self.encoders[0](out)
            skip_connect.append(out)
            out = self.encoders[1](out)
            skip_connect.append(out)
            out = self.encoders[2](out)

            # out = self.encoders[3](out, skip_connect.pop())
            # out = self.encoders[4](out, skip_connect.pop())
            # out = self.encoders[5](out, skip_connect.pop())

            out = self.decoders[0](out, skip_connect.pop())
            out = self.decoders[1](out, skip_connect.pop())
            out = self.decoders[2](out, skip_connect.pop())

            out = self.classifier(out)
            
            return out
    
    return ecotta_swinunet3d(simplified_model, num_blocks, in_channels_per_partition, out_channels_per_partition, spatial_scales_per_partition)


def create_ecotta_module(base_model_ckpt, device, num_classes=2, K=3):
    
    base_model = SwinUnet3d(num_classes=num_classes).to(device)
    
    if os.path.exists(base_model_ckpt):
        print(f"Loading pre-trained SwinUnet3d model from {base_model_ckpt}")
        checkpoint = torch.load(base_model_ckpt, map_location=device, weights_only=True)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print("Pre-trained SwinUnet3d model loaded successfully")
    else:
        print(f"Pre-trained SwinUnet3d model not found at {base_model_ckpt}")
        print("Initializing with random weights")

    simplified_model = simplify_SwinUnet3d(base_model).to(device)
    
    ecotta_model = attach_meta_networks_resunet3D(simplified_model, K=K).to(device)
    
    # frozen original network and train meta-network
    for param in ecotta_model.parameters():
        param.requires_grad = False
    for meta_part in ecotta_model.meta_parts:
        for param in meta_part.parameters():
            param.requires_grad = True

    return ecotta_model, base_model
