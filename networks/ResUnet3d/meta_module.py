import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch import ResUnet3d


"""## TTA Implementation for ResUnet3d """


class SpectralAdapter(nn.Module):
    def __init__(self, in_dim, modes=16):
        super().__init__()
        self.in_dim = in_dim
        self.modes = modes
        
        self.weights1 = nn.Parameter(
            torch.rand(in_dim, in_dim, dtype=torch.cfloat)
        )

        self.weights2 = nn.Parameter(
            torch.rand(in_dim, in_dim, dtype=torch.cfloat)
        )
    def forward(self, x):
        B, C, H, W, D = x.shape
        
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1), norm='ortho')
        x_ft_shifted = torch.fft.fftshift(x_ft, dim=(-3, -2, -1))
        
        h_center, w_center, d_center = H // 2, W // 2, D // 2
        h_start = h_center - self.modes // 2
        h_end = h_start + self.modes
        w_start = w_center - self.modes // 2
        w_end = w_start + self.modes
        d_start = d_center - self.modes // 2
        d_end = d_start + self.modes
        
        out_ft_shifted = torch.einsum("bixyz,io->boxyz", x_ft_shifted, self.weights2)
        
        center_patch = x_ft_shifted[:, :, h_start:h_end, w_start:w_end, d_start:d_end]
        adapted_center = torch.einsum("bixyz,io->boxyz", center_patch, self.weights1)
        
        out_ft_shifted[:, :, h_start:h_end, w_start:w_end, d_start:d_end] = adapted_center
        
        out_ft = torch.fft.ifftshift(out_ft_shifted, dim=(-3, -2, -1))
        x_out = torch.fft.ifftn(out_ft, s=(H, W, D), dim=(-3, -2, -1), norm='ortho')
        
        x_out = x_out.real
        
        return x_out


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
        self.module_list = [self.conv1, self.b1, self.b2, self.b3, self.b4, \
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


class one_part_of_networks3d(nn.Module):
    def __init__(self, in_dim, shape, original_part):
        super().__init__()
        self.in_dim = in_dim
        self.num_proj = in_dim

        self.mode = 0 # 0: inference, 1: warm-up, 2: training
        self.ns = 0
        self.ns_memory = None

        self.original_part = original_part
        # self.meta_part = nn.Sequential(
        #     nn.Conv3d(in_dim, in_dim, kernel_size=1, bias=False),
        #     nn.ReLU()
        # )
        self.meta_part = nn.Sequential(
            SpectralAdapter(in_dim, modes=int(shape*1/2)),
            nn.ReLU()
        )

        self.register_buffer("wb_center", None)
        self.register_buffer("running_mu", None)
        self.register_buffer("running_var", None)

        rand = torch.randn(self.in_dim, self.num_proj)
        rand = rand / rand.norm(dim=1).unsqueeze(1)
        self.register_buffer("rand", rand)

        self.init_as_identity()

    def init_as_identity(self):
        nn.init.eye_(self.meta_part[0].weights1)
        nn.init.eye_(self.meta_part[0].weights2)
        # nn.init.eye_(self.meta_part[0].weight.view(self.in_dim, self.in_dim))

    def forward(self, x):
        _, C, H, W, D = x.shape

        x_orig = self.original_part(x)
        x = self.meta_part(x_orig)
        out = x

        if self.mode == 0: # inference
            return out
        
        rand = self.__getattr__("rand")

        if H > 64 and W > 64 and D > 64:
            x = F.avg_pool3d(x, kernel_size=4, stride=4)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1).squeeze(0) # N, C

        proj = torch.matmul(x, rand).permute(1, 0) # C, N
        sorted_proj, _ = torch.sort(proj, dim=-1)

        cur_mu = x.mean(dim=0)
        cur_var = x.var(dim=0)
        
        if self.mode == 1: # warm-up
            if self.ns == 0:
                self.wb_center = sorted_proj.detach()
                self.running_mu = cur_mu.detach()
                self.running_var = cur_var.detach()
            else:
                self.wb_center = self.wb_center * self.ns / (self.ns + 1) + sorted_proj.detach() / self.ns
                self.running_mu = (self.running_mu * self.ns + cur_mu.detach()) / (self.ns + 1)
                self.running_var = (self.running_var * self.ns + cur_var.detach()) / (self.ns + 1) + \
                    (self.ns / ((self.ns + 1)**2)) * (cur_mu.detach() - self.running_mu)**2
                
        if self.mode == 2: # training
            self.loss_swd = F.l1_loss(sorted_proj, self.wb_center, reduction='mean')
            self.loss_bn = F.l1_loss(self.running_mu, cur_mu, reduction='mean') + \
                            F.l1_loss((self.running_var + 1e-6).sqrt(), (cur_var + 1e-6).sqrt(), reduction='mean')

            self.loss_reg = F.mse_loss(x_orig, out, reduction='mean')

        self.ns += 1

        return out


class tta_resunet3d(nn.Module):
    def __init__(self, simplified_model, out_channels_per_partition, K):
        super().__init__()
        
        self.module_list = simplified_model.module_list
        self.classifier = simplified_model.classifier
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.meta_parts = nn.ModuleList()
        
        shape = [160, 160, 80, 40, 20]
        for i in range(K):
            original_part = self.module_list[i]
            wrapped_part = one_part_of_networks3d(out_channels_per_partition[i], shape[i], original_part)
            self.encoders.append(wrapped_part)
            self.meta_parts.append(wrapped_part.meta_part)

        for decoder in self.module_list[K:]:
            self.decoders.append(decoder)

    def set_mode(self, mode):
        for encoder in self.encoders:
            encoder.mode = mode

    def forward(self, x):
        skip_connect = []

        out = self.encoders[0](x)
        skip_connect.append(out)
        out = self.encoders[1](out)
        skip_connect.append(out)
        out = self.encoders[2](out)
        skip_connect.append(out)
        out = self.encoders[3](out)
        skip_connect.append(out)
        out = self.encoders[4](out)

        out = self.decoders[0](out, skip_connect.pop())
        out = self.decoders[1](out, skip_connect.pop())
        out = self.decoders[2](out, skip_connect.pop())
        out = self.decoders[3](out, skip_connect.pop())

        out = self.classifier(out)
        
        return out


def create_meta_module(base_model_ckpt, device, num_classes=2, K=5):

    base_model = ResUnet3d(resnet='resnet3d34', num_classes=num_classes).to(device)
    
    if os.path.exists(base_model_ckpt):
        print(f"Loading pre-trained ResUNet model from {base_model_ckpt}")
        checkpoint = torch.load(base_model_ckpt, map_location=device, weights_only=True)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print("Pre-trained ResUNet model loaded successfully")
    else:
        print(f"Pre-trained ResUNet model not found at {base_model_ckpt}")
        print("Initializing with random weights")

    simplified_model = simplify_resunet3d(base_model).to(device)
    
    in_channels_per_partition = [3, 64, 64, 128, 256, 512, 256, 128, 64]  # 每个阶段的输入通道
    out_channels_per_partition = [64, 64, 128, 256, 512, 256, 128, 64, 64]  # 每个阶段的输出通道
    spatial_scales_per_partition = [0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2]

    tta_model = tta_resunet3d(simplified_model, out_channels_per_partition, K).to(device)
    
    # frozen original network and train meta-network
    for param in tta_model.parameters():
        param.requires_grad = False
    for meta_part in tta_model.meta_parts:
        for param in meta_part.parameters():
            param.requires_grad = True

    return tta_model
