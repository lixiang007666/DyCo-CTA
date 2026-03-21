import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet3d import resnet3d18, resnet3d34, resnet3d50, resnet3d101, resnet3d152


class UnetBlock3D(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv3d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose3d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm3d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        
        # 确保up_p和x_p在空间维度上对齐
        if up_p.shape[2:] != x_p.shape[2:]:
            # 使用插值方法调整up_p的尺寸以匹配x_p
            up_p = F.interpolate(up_p, size=x_p.shape[2:], mode='trilinear', align_corners=False)
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        # return self.bn(F.relu(cat_p))
        return F.relu(self.bn(cat_p))


class ResUnet3d(nn.Module):
    def __init__(self, resnet='resnet3d34', num_classes=2):
        super().__init__()
        if resnet == 'resnet3d18':
            base_model = resnet3d18
            bottleneck = False
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet3d34':
            base_model = resnet3d34
            bottleneck = False
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet3d50':
            base_model = resnet3d50
            bottleneck = True
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet3d101':
            base_model = resnet3d101
            bottleneck = True
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet3d152':
            base_model = resnet3d152
            bottleneck = True
            feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise Exception('The Resnet3D Model only accept resnet3d18, resnet3d34, resnet3d50, resnet3d101 and resnet3d152!')

        self.encoder = base_model()
        self.num_classes = num_classes

        self.up1 = UnetBlock3D(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock3D(256, feature_channels[2], 128)
        self.up3 = UnetBlock3D(128, feature_channels[1], 64)
        self.up4 = UnetBlock3D(64, feature_channels[0], 64)

        self.seg_head = nn.Sequential(nn.ConvTranspose3d(64, 16, kernel_size=2, stride=2), 
                                      nn.BatchNorm3d(16), 
                                      nn.ReLU(),
                                      nn.Conv3d(16, num_classes, 1)
                                      )

    def forward(self, x):
        sfs = self.encoder(x)
        x = sfs[-1]
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        seg_output = self.seg_head(x)

        return seg_output

