import torch
import torch.nn as nn
import numpy as np



class AdaBN3D(nn.BatchNorm3d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN3D, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3, 4), keepdims=True).detach()
        cur_var = x.var((0, 2, 3, 4), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1, 1)

        moment = torch.tensor(1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1))

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W, D = x.shape

        new_mu, new_var = self.get_mu_var(x)

        cur_mu = x.mean((2, 3, 4), keepdims=True)
        cur_std = x.std((2, 3, 4), keepdims=True)
        self.bn_loss = (
                (new_mu - cur_mu).abs().mean() + (new_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1, 1) + self.bias.view(1, C, 1, 1, 1)
        return new_x


def convert_encoder_to_target_3d(net, norm, start=0, end=5, verbose=True, input_size=128, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(old_norm.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 0:
            # Convert the first conv layer's BatchNorm3d
            net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features, idx, fea_size=input_size // 2)
            idx += 1
        else:
            down_sample = 2 ** (1 + i)

            for j, block in enumerate(layer):
                # Convert block.bn1 and block.bn2
                block.bn1 = convert_norm(block.bn1, norm, block.bn1.num_features, idx, fea_size=input_size // down_sample)
                block.bn2 = convert_norm(block.bn2, norm, block.bn2.num_features, idx, fea_size=input_size // down_sample)
                idx += 1
                if block.downsample is not None:
                    for idx_ds, module in enumerate(block.downsample):
                        if isinstance(module, nn.BatchNorm3d):
                            block.downsample[idx_ds] = convert_norm(module, norm, module.num_features, idx, fea_size=input_size // down_sample)
                            break
                    idx += 1
    return net


def convert_decoder_to_target_3d(net, norm, start=0, end=5, verbose=True, input_size=128, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(old_norm.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [net[0], net[1], net[2], net[3], net[4]]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 4:
            # Handle the last layer separately
            layers[i] = convert_norm(layer, norm, layer.num_features, idx, input_size)
            idx += 1
        else:
            down_sample = 2 ** (4 - i)
            # Convert the BatchNorm3d in UnetBlock3D
            if hasattr(layer, 'bn'):
                layer.bn = convert_norm(layer.bn, norm, layer.bn.num_features, idx, input_size // down_sample)
                idx += 1
    # Update the network with converted layers
    net[0], net[1], net[2], net[3], net[4] = layers[0], layers[1], layers[2], layers[3], layers[4]
    return net