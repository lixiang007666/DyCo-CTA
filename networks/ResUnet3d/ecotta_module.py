import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch import ResUnet3d


"""## EcoTTA Implementation for ResUnet3d """

class simplify_resunet3d(nn.Module):
    def __init__(self, model):
        super().__init__()
        # encoder
        self.conv1 = nn.Sequential(model.encoder.conv1,
                                   model.encoder.bn1, 
                                   model.encoder.relu)
        self.b1 = nn.Sequential(model.encoder.maxpool, 
                                model.encoder.layer1)
        self.b2 = model.encoder.layer2
        self.b3 = model.encoder.layer3
        self.b4 = model.encoder.layer4
        self.up1 = model.up1
        self.up2 = model.up2
        self.up3 = model.up3
        self.up4 = model.up4
        self.module_list = [self.b1, self.b2, self.b3, self.b4, \
                        self.up1, self.up2, self.up3, self.up4]
        self.classifier = model.seg_head

    def forward(self, x):
        out = self.conv1(x)
        out = self.b1(out)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.up1(out)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)
        out = self.classifier(out)
        return out

class conv_block3D(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, stride=1):
        super().__init__()
        if stride > 1:
            self.conv = nn.Conv3d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        elif stride == 1:
            self.conv = nn.Conv3d(in_plane, out_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_plane)
    
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
            out2 = self.meta_part.meta_bn(out1) if hasattr(self.meta_part, 'meta_bn') else out1
            out3 = self.meta_part(x)
            out = out2 + out3
        else:
            x = x.detach()
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1) if hasattr(self.meta_part, 'meta_bn') else out1
            out3 = self.meta_part(x)
            out = out2 + out3
            loss = nn.L1Loss(reduction='none')
            self.btsloss = loss(out, out1.detach()).mean()
        return out


def attach_meta_networks_resunet3d(simplified_model, K=5):
    if K == 4:
        num_blocks = [1, 1, 1, 1]
    else:
        raise ValueError("K should be 4")
    
    in_channels_per_partition = [64, 64, 128, 256]  # 每个阶段的输入通道
    out_channels_per_partition = [64, 128, 256, 512]  # 每个阶段的输出通道
    spatial_scales_per_partition = [2, 2, 2, 2]
    
    
    class ecotta_resunet3d(nn.Module):
        def __init__(self, simplified_model, num_blocks, in_channels_per_partition, out_channels_per_partition, spatial_scales_per_partition):
            super().__init__()
            
            self.conv1 = simplified_model.conv1
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
            skip_connect.append(out)
            out = self.encoders[3](out)

            out = self.decoders[0](out, skip_connect.pop())
            out = self.decoders[1](out, skip_connect.pop())
            out = self.decoders[2](out, skip_connect.pop())
            out = self.decoders[3](out, skip_connect.pop())

            out = self.classifier(out)
            
            return out
    
    return ecotta_resunet3d(simplified_model, num_blocks, in_channels_per_partition, out_channels_per_partition, spatial_scales_per_partition)

def create_ecotta_module(base_model_ckpt, device, num_classes=2, K=5):

    base_model = ResUnet3d(resnet='resnet3d34', num_classes=num_classes, convert=False).to(device)
    
    if os.path.exists(base_model_ckpt):
        print(f"Loading pre-trained ResUNet model from {base_model_ckpt}")
        checkpoint = torch.load(base_model_ckpt, map_location=device, weights_only=True)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print("Pre-trained ResUNet model loaded successfully")
    else:
        print(f"Pre-trained ResUNet model not found at {base_model_ckpt}")
        print("Initializing with random weights")

    simplified_model = simplify_resunet3d(base_model).to(device)
    
    ecotta_model = attach_meta_networks_resunet3d(simplified_model, K=K).to(device)
    
    # frozen original network and train meta-network
    for param in ecotta_model.parameters():
        param.requires_grad = False
    for meta_part in ecotta_model.meta_parts:
        for param in meta_part.parameters():
            param.requires_grad = True

    return ecotta_model, base_model
