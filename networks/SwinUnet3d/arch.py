import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Union, List
from timm.layers import trunc_normal_


class CyclicShift3D(nn.Module):
    def __init__(self, displacement):
        super().__init__()

        assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
        if type(displacement) is int:
            displacement = np.array([displacement, displacement, displacement])
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[1], self.displacement[2]), dims=(1, 2, 3))


class FeedForward3D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout: float = 0.0):
        super().__init__()

        self.prenorm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        input = x.clone()
        x = self.prenorm(x)
        x = self.net(x)
        x = self.drop(x)
        return x + input


def create_mask3D(window_size: Union[int, List[int]], displacement: Union[int, List[int]],
                  x_shift: bool, y_shift: bool, z_shift: bool):
    assert len(window_size) == len(displacement)
    for i in range(len(window_size)):
        assert 0 < displacement[i] < window_size[i], \
            f'在第{i}轴的偏移量不正确，维度包括X(i=0)，Y(i=1)和Z(i=2)'

    mask = torch.zeros(window_size[0] * window_size[1] * window_size[2],
                       window_size[0] * window_size[1] * window_size[2])  # (wx*wy*wz, wx*wy*wz)
    mask = rearrange(mask, '(x1 y1 z1) (x2 y2 z2) -> x1 y1 z1 x2 y2 z2',
                     x1=window_size[0], y1=window_size[1], x2=window_size[0], y2=window_size[1])

    x_dist, y_dist, z_dist = displacement[0], displacement[1], displacement[2]

    if x_shift:
        mask[-x_dist:, :, :, :-x_dist, :, :] = float('-inf')
        mask[:-x_dist, :, :, -x_dist:, :, :] = float('-inf')

    if y_shift:
        mask[:, -y_dist:, :, :, :-y_dist, :] = float('-inf')
        mask[:, :-y_dist, :, :, -y_dist:, :] = float('-inf')

    if z_shift:
        mask[:, :, -z_dist:, :, :, :-z_dist] = float('-inf')
        mask[:, :, :-z_dist, :, :, -z_dist:] = float('-inf')

    mask = rearrange(mask, 'x1 y1 z1 x2 y2 z2 -> (x1 y1 z1) (x2 y2 z2)')
    return mask


class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted: bool, window_size: Union[int, List[int]]):
        super().__init__()

        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted

        self.prenorm = nn.LayerNorm(dim)

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift3D(-displacement)
            self.cyclic_back_shift = CyclicShift3D(displacement)
            self.x_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=True, y_shift=False, z_shift=False), requires_grad=False)
            self.y_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=False, y_shift=True, z_shift=False), requires_grad=False)
            self.z_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=False, y_shift=False, z_shift=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        input = x.clone()

        _, orig_x, orig_y, orig_z, _ = x.shape

        w_x, w_y, w_z = self.window_size
        pad_x = (w_x - orig_x % w_x) % w_x
        pad_y = (w_y - orig_y % w_y) % w_y
        pad_z = (w_z - orig_z % w_z) % w_z

        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            x = F.pad(x, (0, 0, 0, pad_z, 0, pad_y, 0, pad_x))
        
        x = self.prenorm(x)

        if self.shifted:
            x = self.cyclic_shift(x)

        _, X, Y, Z, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_x = X // self.window_size[0]
        nw_y = Y // self.window_size[1]
        nw_z = Z // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d',
                                h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2]), qkv)

        weights = einsum('b h n i d, b h n j d -> b h n i j', q, k) * self.scale

        if self.shifted:
            # 将x轴的窗口数量移至尾部，便于和x轴上对应的mask叠加，下同
            weights = rearrange(weights, 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j', n_x=nw_x, n_y=nw_y)
            weights[:, :, :, :, -1] += self.x_mask

            weights = rearrange(weights, 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j')
            weights[:, :, :, :, -1] += self.y_mask

            weights = rearrange(weights, 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j')
            weights[:, :, :, :, -1] += self.z_mask

            weights = rearrange(weights, 'b h n_y n_z n_x i j -> b h (n_x n_y n_z) i j')

        attn = self.softmax(weights)
        out = einsum('b h n i j, b h n j d -> b h n i d', attn, v)

        # nw_x 表示x轴上窗口的数量 , nw_y 表示 y轴上窗口的数量，nw_Z表示z轴上窗口的数量
        # w_x 表示 x_window_size, w_y 表示 y_window_size， w_z表示z_window_size
        #                     b 3  (8,8,8)         （7,  7,  7） 96 -> b  56          56          56        288
        out = rearrange(out, 'b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)',
                        h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2], nw_x=nw_x, nw_y=nw_y, nw_z=nw_z)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            out = out[:, :orig_x, :orig_y, :orig_z, :]

        return out + input


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]], dropout: float = 0.0):
        super().__init__()
        self.attention_block = WindowAttention3D(dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size)
        self.mlp_block = FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class Norm(nn.Module):
    def __init__(self, dim, channel_first: bool = True):
        super(Norm, self).__init__()
        if channel_first:
            self.net = nn.Sequential(
                Rearrange('b c h w d -> b h w d c'),
                nn.LayerNorm(dim),
                Rearrange('b h w d c -> b c h w d')
            )
        else:
            self.net = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.net(x)
        return x


class ScaleBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, scale_factor, downscale=True):
            super(ScaleBlock, self).__init__()

            if downscale:
                self.scale_net = nn.Sequential(
                    nn.Conv3d(in_dim, hidden_dim, kernel_size=scale_factor, stride=scale_factor),
                    Norm(dim=hidden_dim),
                )
            else:
                self.scale_net = nn.Sequential(
                    nn.ConvTranspose3d(in_dim, hidden_dim, kernel_size=scale_factor, stride=scale_factor),
                    Norm(hidden_dim),
                )
    
    def forward(self, x):
        return self.scale_net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            Norm(dim=hidden_dim),
            nn.PReLU(),

            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            Norm(dim=hidden_dim),
            nn.PReLU()
        )

    def forward(self, x):
        # (B, C, H, W, D)
        x1 = x.clone()
        x = self.net(x) * x1
        return x


class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers, scale_factor, num_heads, head_dim, window_size: Union[int, List[int]], 
                 dropout: float=0.0, downscale: bool=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.scale_net = ScaleBlock(in_dim=in_dim, hidden_dim=hidden_dim, scale_factor=scale_factor, downscale=downscale)

        self.conv_block = ConvBlock(in_dim=hidden_dim, hidden_dim=hidden_dim)

        self.re1 = Rearrange('b c h w d -> b h w d c')
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.swin_layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dim, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dim * 4,
                            shifted=False, window_size=window_size, dropout=dropout),
                SwinBlock3D(dim=hidden_dim, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dim * 4,
                            shifted=True, window_size=window_size, dropout=dropout)
            ]))
        self.re2 = Rearrange('b  h w d c -> b c h w d')

    def forward(self, x):
        # scaling
        x = self.scale_net(x)

        # conv
        x1 = self.conv_block(x)

        # transformer
        x2 = self.re1(x)
        for regular_block, shifted_block in self.swin_layers:
            x2 = regular_block(x2)
            x2 = shifted_block(x2)
        x2 = self.re2(x2)

        x = x1 + x2
        return x
    

class Merge(nn.Module):
    def __init__(self, fn, in_dim, hidden_dim, **kwargs):
        super(Merge, self).__init__()
        assert hidden_dim % 2 == 0

        out_dim = hidden_dim // 2

        self.sc_conv = nn.Conv3d(hidden_dim, out_dim, 1)
        self.fn = fn(in_dim, out_dim, **kwargs)
        self.norm = Norm(hidden_dim)

    def forward(self, x, enc_x):
        enc_x = self.sc_conv(enc_x)
        x = self.fn(x)
        assert x.shape == enc_x.shape
        cat = torch.cat([x, enc_x], dim=1)
        return self.norm(F.relu(cat))


class SwinUnet3d(nn.Module):
    def __init__(self, hidden_dim=(64, 256, 512, 1024), layers=(2, 2, 2, 2), heads=(2, 4, 6, 8), in_dim=1, num_classes=2, 
                 head_dim=32, window_size: Union[int, List[int]] = 7, scale_factors=(2, 2, 2, 2), dropout: float = 0.0):
        super().__init__()

        self.dsf = scale_factors

        assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])

        self.enc = nn.ModuleList()
        in_dim = in_dim
        for i in range(len(scale_factors)):
            self.enc.append(Block(in_dim=in_dim, hidden_dim=hidden_dim[i], layers=layers[i], 
                              scale_factor=scale_factors[i], num_heads=heads[i], head_dim=head_dim, 
                              window_size=window_size, dropout=dropout))
            in_dim = hidden_dim[i]

        self.dec = nn.ModuleList()
        self.converge = nn.ModuleList()
        for i in reversed(range(len(scale_factors) - 1)):
            in_dim = hidden_dim[i+1]
            self.dec.append(Merge(Block, in_dim=in_dim, hidden_dim=hidden_dim[i], layers=layers[i], 
                              scale_factor=scale_factors[i+1], num_heads=heads[i], head_dim=head_dim, 
                              window_size=window_size, dropout=dropout, downscale=False))

        self.classifier = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim[0], 8, kernel_size=scale_factors[0], stride=scale_factors[0]),
            Norm(8),
            nn.PReLU(),
            nn.Conv3d(8, num_classes, kernel_size=1)
        )

        self.init_weight()

    def forward(self, x):
        skip_connect = []

        for block in self.enc:
            x = block(x)
            skip_connect.append(x)

        skip_connect.pop()
        for block in self.dec:
            x = block(x, skip_connect.pop())

        out = self.classifier(x)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)