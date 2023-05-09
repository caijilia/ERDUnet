from re import S
from turtle import forward
from xml.dom.minidom import Identified
import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class DRA(nn.Module):
    def __init__(self, in_ch=64, class_num=2, value=0.5, simple=1, large=0):
        super(DRA, self).__init__()
        self.threshold = value
        self.s_mode = simple
        self.l_mode = large
        self.sigmoid = nn.Sigmoid()
        self.add_learn = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )
        self.to_class_1 = nn.Conv2d(in_channels=in_ch, out_channels=class_num, kernel_size=1)
        self.to_class_2 = nn.Conv2d(in_channels=in_ch, out_channels=class_num, kernel_size=1)
        self.region_learn = nn.Sequential(
            nn.Conv2d(in_channels=class_num, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
    
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        if self.s_mode == 1:
            x1_in       = 1*(x1 > self.threshold) 
            x2_in       = 1*(x2 > self.threshold) 
            edge_diff   = torch.abs(x1_in - x2_in)          
            x           = (edge_diff) * x1 + x1 
            x           = self.add_learn(x)               
        elif self.l_mode == 1:
            x1_in       = self.to_class_1(x1)
            x2_in       = self.to_class_2(x2)
            x1_in       = 1*(x1_in > self.threshold) 
            x2_in       = 1*(x2_in > self.threshold) 
            edge_diff   = torch.abs(x1_in - x2_in) * 1.000000
            edge_diff   = self.region_learn(edge_diff) 
            x           = self.sigmoid(edge_diff) * x1 + x1
            x           = self.add_learn(x) 
        return x

class IRE(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0):
        super(IRE, self).__init__()
        self.fc1        = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/rate), kernel_size=1)
        self.relu       = nn.ReLU(inplace=True)
        self.fc2        = nn.Conv2d(in_channels=int(in_ch/rate), out_channels=in_ch, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()

        self.compress   = ChannelPool()
        self.spatial    = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        self.fc3        = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/rate), kernel_size=1)
        self.fc4        = nn.Conv2d(in_channels=int(in_ch/rate), out_channels=in_ch, kernel_size=1)

        self.ch_use     = only_ch 
    
    def forward(self, x):
        x_in = x                
        x = x.mean((2, 3), keepdim=True) 
        x = self.fc1(x)         
        x = self.relu(x)        
        x = self.fc2(x)         
        x = self.sigmoid(x) * x_in 
        if self.ch_use == 1:
            return x
        elif self.ch_use == 0:
            x = x

        s_in = x                    
        s = self.compress(x)        
        s = self.spatial(s)         
        s = self.sigmoid(s) * s_in  

        c_in = s                    
        c = self.fc3(s)             
        c = self.relu(c)
        c = self.fc4(c)             
        c = self.sigmoid(c) * c_in  
    
        return c

class MRA(nn.Module):
    def __init__(self, c1_in_channels=64, c2_in_channels=128, c3_in_channels=256, embedding_dim=256, drop_rate=0.2, classes=2):
        super(MRA, self).__init__()
        self.conv_c1 = nn.Conv2d(in_channels=c1_in_channels, out_channels=embedding_dim, kernel_size=1)
        self.down_1  = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1)
        self.conv_c2 = nn.Conv2d(in_channels=c2_in_channels, out_channels=embedding_dim, kernel_size=1)
        self.down_2  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim*3, out_channels=embedding_dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU()
        )
        self.drop = nn.Dropout2d(drop_rate)
        self.edge = DRA(in_ch=256, class_num=classes)
    
    def forward(self, inputs):
        c1, c2, c3 = inputs

        c1_ = self.conv_c1(c1) 
        c1_ = self.down_1(c1_) 

        c2_ = self.conv_c2(c2) 
        c2_ = self.down_2(c2_) 

        c3_ = c3 

        c_fuse = self.conv_fuse(torch.cat([c1_, c2_, c3_], dim=1)) 
        x = self.drop(c_fuse) + self.edge(c2_, c1_) + self.edge(c3_, c2_)

        return x

#-------------------------------------------------------------------------------------------------------
class DET(nn.Module):
    def __init__(self, input_dim=64, out_dim=64, conv_rate=8, simple=0):
        super(DET, self).__init__()
        self.simple = simple
        self.conv_0 = nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=1)
        self.conv_1 = nn.Conv2d(in_channels=input_dim, out_channels=int(out_dim / conv_rate), kernel_size=1, padding=0, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=int(out_dim / conv_rate), out_channels=int(out_dim / conv_rate), kernel_size=1, padding=0, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=int(out_dim / conv_rate), out_channels=out_dim, kernel_size=1, padding=0, stride=1)
        self.gelu   = nn.GELU()
        self.norm   = nn.BatchNorm2d(out_dim)
        # self.cspatt = IRE(in_ch=out_dim, rate=4)
        self.conv_0_s = nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # identity
        if self.simple == 0:
            x_0 = self.conv_0(x)
            x_0 = self.norm(x_0)
        else:
            x_0 = self.conv_0_s(x)
            x_0 = self.norm(x_0)
        # learn_path
        x_1 = self.gelu(self.conv_1(x))
        x_2 = self.gelu(self.conv_2(x_1))
        x_3 = self.conv_3(x_2)
        # fusion
        x_incep = self.gelu(self.norm(x_0 + x_3))
        return x_incep
#-------------------------------------------------------------------------------------------------------
class PFC(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.pointwise(x)
        return x

class CEE(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_chans=64, embed_dim=64, smaller=0, use_att=0):
        super().__init__()
        self.att_use = use_att
        self.att = IRE(in_ch=embed_dim, rate=4, only_ch=0)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2)),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.proj_c = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True, groups=embed_dim),
            nn.GELU()
        )
        self.fc0 = Conv(embed_dim, embed_dim, 3, bn=True, relu=True)
        self.dwconv_1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU()
        )
        self.use_small_conv = smaller
        self.dwconv_2 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.turn_channel = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_pe = self.proj(x)
        if self.att_use == 1:
            x_pe = self.att(x_pe)
        x_pe_conv = self.proj_c(x)
        x_PE = x_pe.flatten(2).transpose(1, 2) 
        x_PE = self.norm(x_PE)
        x_po = self.dwconv(x_pe).flatten(2).transpose(1, 2) 
        x_0  = torch.transpose((x_PE + x_po), 1, 2).view(b, x_pe.shape[1], int(h/2), int(w/2))
        x_0  = self.fc0(x_0) 
        x_1  = x_0 
        if self.use_small_conv == 1:
            x_1_ = self.dwconv_2(torch.cat([x_1, x_pe_conv], dim=1))
            x_1_ = self.turn_channel(torch.cat([x_1_, x_pe], dim=1)).flatten(2).transpose(1, 2)
            x_out  = x_1_ + x_PE
            return x_out
        else:
            x_1_ = self.dwconv_1(x_1) 
            x_1_ = self.turn_channel(torch.cat([x_1, x_pe], dim=1)).flatten(2).transpose(1, 2)
            x_out  = self.fc1(x_1_) + x_PE
            return x_out

class ERSUnet(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, in_h=192, in_w=256, normal_init=True, pretrained=False, bound=True, single_object=True):
        super(ERSUnet, self).__init__()
        self.model_h = in_h
        self.model_w = in_w

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_1 = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )


        self.drop = nn.Dropout2d(drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        #---------------------------------------------------------------------------
        self.head = MRA(c1_in_channels=64, c2_in_channels=128, c3_in_channels=256, embedding_dim=256, classes=num_classes)
        self.att_0 = IRE(in_ch=256, rate=4, only_ch=0)
        self.att_1 = IRE(in_ch=256, rate=4, only_ch=0)
        
        self.toshow_p0  = nn.Sequential(nn.Identity()) 
        self.toshow_p1  = nn.Sequential(nn.Identity()) 
        self.toshow_p2  = nn.Sequential(nn.Identity()) 
        self.toshow_p3  = nn.Sequential(nn.Identity()) 
        
        self.toshow_bi  = nn.Identity()
        self.toshow_att = nn.Identity()
        self.out_norm_0 = nn.BatchNorm2d(256)
        self.out_norm_1 = nn.BatchNorm2d(256)

        self.lowest_layer_head = PFC(in_ch=3, channels=16, kernel_size=7)
        self.base_pe_0 = CEE(patch_size=3, stride=2, in_chans=16, embed_dim=32, smaller=1) 
        self.base_pe_1 = CEE(patch_size=3, stride=2, in_chans=32, embed_dim=64, smaller=1) 
        self.base_pe_2 = CEE(patch_size=3, stride=2, in_chans=64, embed_dim=128, smaller=1, use_att=0) 
        self.base_pe_3 = CEE(patch_size=3, stride=2, in_chans=128, embed_dim=256, smaller=1, use_att=0) 

        self.out_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.out_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.out_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.out_1_skip = nn.Sequential(
            IRE(in_ch=128, rate=4, only_ch=0),
            nn.BatchNorm2d(128)
        )
        self.out_2_skip = nn.Sequential(
            IRE(in_ch=64, rate=4, only_ch=0),
            nn.BatchNorm2d(64)
        )
        self.out_3_skip = nn.Sequential(
            nn.GELU()
        )

        self.skip_0_1 = DRA(in_ch=256, class_num=num_classes, value=0.5, simple=1, large=0)
        self.skip_1_1 = DRA(in_ch=128, class_num=num_classes, value=0.5, simple=0, large=1)
        self.skip_2_1 = DRA(in_ch=64, class_num=num_classes, value=0.5, simple=0, large=1)
        self.skip_3   = DRA(in_ch=32, class_num=num_classes, value=0.5, simple=1, large=0)

        self.out_gelu_0 = nn.GELU()
        self.out_gelu_1 = nn.GELU()
        self.out_gelu_2 = nn.GELU()
        self.out_gelu_3 = nn.GELU()

        self.layer_0_1 = nn.Sequential(
            DET(input_dim=32, out_dim=64, conv_rate=2, simple=0),
            nn.Identity()
        )
        self.down_0_1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.layer_1_1 = nn.Sequential(
            DET(input_dim=64, out_dim=128, conv_rate=2, simple=1),
            nn.Identity()
        )
        self.down_1_1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        #---------------------------------------------------------------------------

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        x_b_0_before = self.lowest_layer_head(imgs) 

        input_for_head = []
        x_b_0 = self.base_pe_0(x_b_0_before) 
        x_b_0 = self.dropout(x_b_0)
        x_b_0 = torch.transpose(x_b_0, 1, 2).view(x_b_0.shape[0], -1, (self.model_h // 2), (self.model_w // 2))
        x_b_0 = self.toshow_p0(x_b_0)

        x_b_2_ = self.base_pe_1(x_b_0) 
        x_b_2_ = self.dropout(x_b_2_)
        x_b_2  = torch.transpose(x_b_2_, 1, 2).view(x_b_2_.shape[0], -1, (self.model_h // 4), (self.model_w // 4))
        x_b_2  = self.toshow_p1(x_b_2)
        input_for_head.append(x_b_2)

        x_b_1_ = self.base_pe_2(x_b_2) 
        x_b_1_ = self.dropout(x_b_1_) 
        x_b_1  = torch.transpose(x_b_1_, 1, 2).view(x_b_1_.shape[0], -1, (self.model_h // 8), (self.model_w // 8)) 
        x_b_1  = self.toshow_p2(x_b_1)
        input_for_head.append(x_b_1)
        
        x_b_ = self.base_pe_3(x_b_1) 
        x_b_ = self.dropout(x_b_)
        x_b  = torch.transpose(x_b_, 1, 2).view(x_b_.shape[0], -1, (self.model_h // 16), (self.model_w // 16))
        x_b  = self.toshow_p3(x_b)
        input_for_head.append(x_b)

        x_skip_0_1 = self.layer_0_1(x_b_0) 
        x_skip_0_1 = self.down_0_1(x_skip_0_1) 
        x_skip_1_1 = self.layer_1_1(x_skip_0_1)
        x_skip_1_1 = self.down_1_1(x_skip_1_1) 

        x_h = self.head(input_for_head)
        x_h = self.toshow_bi(x_h)

        x_h_a = self.att_0(x_h)
        x_h_a = self.toshow_att(x_h_a)
        x_h_a = self.out_norm_0(x_h_a)
        
        x_c_a = self.att_1(x_h_a + x_b)
        x_c_a = self.out_norm_1(x_c_a)
        x_c   = self.out_gelu_0(self.skip_0_1(x_c_a, x_b) + x_c_a)

        x_out = self.out_1(x_c)
        x_out = F.interpolate(x_out, scale_factor=2, mode='bilinear') 
        x_out_a = self.out_1_skip(x_out + x_skip_1_1) 
        x_out = self.out_gelu_1(self.skip_1_1(x_out_a, x_b_1) + x_out_a)

        x_out_1 = self.out_2(x_out)
        x_out_1 = F.interpolate(x_out_1, scale_factor=2, mode='bilinear') 
        x_out_1_a = self.out_2_skip(x_out_1 + x_skip_0_1) 
        x_out_1 = self.out_gelu_2(self.skip_2_1(x_out_1_a, x_b_2) + x_out_1_a)

        x_out_2 = self.out_3(x_out_1)
        x_out_2 = F.interpolate(x_out_2, scale_factor=2, mode='bilinear') 
        x_out_2_a = self.out_3_skip(x_out_2 + x_b_0)
        x_out_2 = self.out_gelu_3(self.skip_3(x_out_2_a, x_b_0) + x_out_2_a)

        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear') 
        map_1 = F.interpolate(self.final_1(x_b), scale_factor=16, mode='bilinear') 
        map_2 = F.interpolate(self.final_2(x_out_2), scale_factor=2, mode='bilinear') 
        
        return map_x, map_1, map_2

    def init_weights(self):
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.head.apply(init_weights)
        self.att_0.apply(init_weights)
        self.att_1.apply(init_weights)
        self.lowest_layer_head.apply(init_weights)
        self.out_1.apply(init_weights)
        self.out_2.apply(init_weights)
        self.out_3.apply(init_weights)
        self.out_1_skip.apply(init_weights)
        self.out_2_skip.apply(init_weights)
        self.out_3_skip.apply(init_weights)
        self.skip_0_1.apply(init_weights)
        self.skip_1_1.apply(init_weights)
        self.skip_2_1.apply(init_weights)
        self.skip_3.apply(init_weights)
        self.layer_0_1.apply(init_weights)
        self.layer_1_1.apply(init_weights)
        self.base_pe_0.apply(init_weights)
        self.base_pe_1.apply(init_weights)
        self.base_pe_2.apply(init_weights)
        self.base_pe_3.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.GELU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

