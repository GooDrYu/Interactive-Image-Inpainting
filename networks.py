import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn


##############################
# External spatial attention
##############################

class Ex_KV(nn.Module):
    def __init__(self, dim, mlp_factor, bias_qkv=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_factor, 1, bias=bias_qkv),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1, bias=bias_qkv),
        )

    def forward(self, x):
        return self.net(x)


class ESP_Attn(nn.Module):
    def __init__(self, in_channels, in_width, in_height, token_facter):
        super().__init__()
        self.height = in_height
        self.width = in_width
        self.channels = in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.ex_k = Ex_KV(in_height, token_facter, bias_qkv=False)
        self.ex_v = Ex_KV(in_width, token_facter, bias_qkv=False)

    def forward(self, x):
        b = x.shape[0]
        x = x.permute(0, 3, 2, 1)  # b c h w -> b w h c
        x = self.norm(x)
        x = x.view(b * self.width, self.height, self.channels)  # b w h c -> (b w) h c
        x = self.ex_k(x)
        x = x.view(b, self.width, self.height, self.channels).transpose(1, 2).flatten(0, 1)  # (b w) h c -> (b h) w c
        x = self.ex_v(x)
        x = x.view(b, self.height, self.width, self.channels).permute(0, 3, 1, 2)  # (b h) w c -> b c h w
        return x


###################
###   Generator
###################

class Generator(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(Generator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        input_dim = config['input_dim']
        cnum = config['ngf']

        self.espa = ESP_Attn(cnum*4, 64, 64, token_facter = 2)
        self.convc = nn.Conv2d(input_dim, cnum*4, 1,bias=False)
        
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        context = x.clone()  
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
            
        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        
        # ESPA
        fin = x.clone()
        h, w = fin.shape[2], fin.shape[3]
        subcontext = F.interpolate(context, (h, w), mode='bilinear', align_corners = False)
        subcontext = self.convc(subcontext)
        submask = F.interpolate(mask, (h, w), mode='nearest') 
        espa_in = fin * submask + subcontext * (1. - submask)
        espa_out = self.espa(espa_in)
        espa_out = espa_out* submask + subcontext * (1. - submask)
        
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = x + espa_out
        x = self.conv11(x)
        x = self.conv12(x)

        out1=x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1,out1

########################
# patchGAN Discriminator
########################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
              *discriminator_block(in_channels , 64, normalization=True),
              *discriminator_block(64, 128),
              *discriminator_block(128, 256),
              *discriminator_block(256, 512),
              nn.ZeroPad2d((1, 0, 1, 0)),
              nn.Conv2d(512, 1, 4, padding=1, bias=False)
          )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
         # 256 -> 1*1*16*16 tensor  128 -> 1 1 8 8 
        return self.model(img)



def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


################
#####   test
################

if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils.tools import get_config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                        help="training configuration")
    args = parser.parse_args()
    config = get_config(args.config)
    g=Generator(config['netG'], False,config['gpu_ids'])
    input = torch.randn(1,3,256,256)
    mask = torch.randn(1,1,256,256)
    out=g(input,mask)
    print(out.shape)