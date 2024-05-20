# from curses import KEY_IC
from audioop import lin2adpcm
from cmath import tanh
from email.policy import strict
from multiprocessing import pool
from multiprocessing.spawn import import_main_path
from re import A, S
from telnetlib import SE
from tokenize import group
from turtle import st
import math
from xmlrpc.client import INVALID_ENCODING_CHAR
import PIL
import time
from cv2 import norm
from importlib_metadata import re
from numpy import identity, reshape
from sklearn.model_selection import GroupShuffleSplit
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import _adaptive_avg_pool2d, conv2d, nn, strided
import torch.nn.init as init
import math
import numpy as np

l1 = 1
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class MLP_Block1(nn.Module):
    def __init__(self, img_channels, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, dim)
        self.GELU = GELU()
        self.Dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(img_channels+1, img_channels+1, kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(img_channels+1)
    def forward(self, x):

        x = self.Linear1(x)
        x = self.GELU(x)
        x = self.Dropout(x)
        x = self.Linear2(x)
        x = self.Dropout(x)

        return x

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, dim)
        self.GELU = GELU()
        self.Dropout = nn.Dropout(dropout)
    def forward(self, x):

        x = self.Linear1(x)
        x = self.GELU(x)
        x = self.Dropout(x)
        x = self.Linear2(x)
        x = self.Dropout(x)

        return x

class Attention1(nn.Module):

    def __init__(self, dim, img_channels,heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.GELU = GELU()

        # self.FC = nn.Linear()
        self.conv = nn.Conv2d(img_channels+1, img_channels+1, kernel_size=1)
        self.bn = nn.BatchNorm2d(img_channels+1)

    def forward(self, x, mask=None):       # x=(64,5,64)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions  
    
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        # print(attn.shape)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.GELU = GELU()

        # self.FC = nn.Linear()
        self.conv = nn.Conv2d(dim+1, dim+1, kernel_size=1)

    def forward(self, x, mask=None):       # [64, 65, 64]

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions    
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out1 = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax

        out = rearrange(out1, 'b h n d -> b n (h d)') 
        # k1 = rearrange(k, 'b h n d -> b n (h d)')
        q1 = self.nn1(out)
        k1 = self.nn1(out)
        v1 = self.nn1(out)
        # print(q1.shape)
        b,n,h = q1.size()
        q1 = q1.view(b,8,n,8)
        k1 = k1.view(b,8,n,8)
        v1 = v1.view(b,8,n,8)


        dots1 = torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale
        mask_value = -torch.finfo(dots1.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots1.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots1.masked_fill_(~mask, float('-inf'))
            del mask

        attn1 = dots1.softmax(dim=-1) 
        out = torch.einsum('bhij,bhjd->bhid', attn1, v1)

        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        # print(out.shape)
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.attention = Attention(dim, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP_Block(dim, mlp_dim, dropout=dropout)

    def forward(self, x1, mask=None):
        # for attention, mlp in self.layers:
        identity = x1         # (64,65,64)
        x1 = self.norm(x1)
        x1 = self.attention(x1, mask=mask)  # go to attention   [64, 65, 64]
        x1 = x1 + identity
        x22 = self.norm(x1)
        x22 = self.mlp(x22)  # go to MLP_Block
        x = x22 + x1
        return x1

class Transformer1(nn.Module):
    def __init__(self, dim,img_channels,depth, heads, mlp_dim, dropout):
        super().__init__()
        self.attention = Attention1(dim,img_channels, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP_Block1(img_channels, dim, mlp_dim, dropout=dropout)

    def forward(self, x1, mask=None):
        # for attention, mlp in self.layers:
        identity = x1         # (64,65,64)
        # print(identity.shape)
        x1 = self.norm(x1)
        x1 = self.attention(x1, mask=mask)  # go to attention   [64, 65, 64]
        # print(x1.shape)
        x1 = x1 + identity
        x22 = self.norm(x1)
        x22 = self.mlp(x22)  # go to MLP_Block
        x = x22 + x1
        return x1

BATCH_SIZE_TRAIN = 32
NUM_CLASS = 16

def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class S2FTNet(nn.Module):
    def __init__(self, xy, BAND, img_channels,band=30, num_classes=16, num_tokens=30, dim=64, depth=1, 
                heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1,reduction_ratio=10, pool_types=['avg']):
        super(S2FTNet, self).__init__()

        self.name = 'S2FTNet'

        global l 
        self.L = num_tokens
        self.cT = dim

        self.pos_embedding1 = nn.Parameter(torch.empty(1, (img_channels + 1), dim))

        self.pos_embedding = nn.Parameter(torch.empty(1, (dim + 1), dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, 192))
        self.dropout = nn.Dropout(emb_dropout)
        # self.l = torch.nn.parameter()

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.transformer1 = Transformer1(dim,img_channels, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.n = nn.Linear(64*4, 128)
        self.nn = nn.Linear(200, 128)
        # self.nn = nn.Linear(128, 256)
        self.nnn = nn.Linear(128, num_classes)
        self.patch_to_embedding = nn.Linear(169, dim)
        self.patch_to_embedding1 = nn.Linear(81, dim)
        self.patch_to_embedding2 = nn.Linear(25, dim)
        # self.patch_to_embedding5 = nn.Linear(3, dim)
        self.patch_to_embedding4 = nn.Linear(3, dim)

        self.dropout5 = nn.Dropout(emb_dropout)
        # self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128)
        )


        self.pool1 = nn.AdaptiveAvgPool2d((9,9))
        self.pool2 = nn.AdaptiveAvgPool2d((5,5))

        self.pool3 = nn.AdaptiveAvgPool2d((13,13))
        self.pool4 = nn.AdaptiveAvgPool2d((6,6))
        self.pool5 = nn.AdaptiveAvgPool2d((3,3))


        self.nn0 = nn.Linear(64*3, 64)
        self.nn1 = nn.Linear(64, 64*3)
        self.sigmoid = nn.Sigmoid()

        self.FC1 = nn.Linear(img_channels, 256)
        self.BN1 = nn.BatchNorm1d(256)
        self.FC2 = nn.Linear(256, 128)
        self.FC3 = nn.Linear(200, 64*1)
        self.att = nn.Sigmoid()

        
 
        self.kongdong1 = nn.Conv3d(1, 64, kernel_size=(1,1,7),stride=(1,1,1),padding=(0,0,2))
        self.batch_normkongdong11 = nn.Sequential(
                                    nn.BatchNorm3d(64,  eps=0.001, momentum=0.1, affine=True)) # 动量默认值为0.1)
        self.relu1 = nn.ReLU()
        self.kongdong2 = nn.Conv3d(32,64,kernel_size=(1,1,7),stride=(1,1,1),padding=(0,0,2))
        self.batch_normkongdong12 = nn.Sequential(
                                    nn.BatchNorm3d(64,  eps=0.001, momentum=0.1, affine=True)) # 动量默认值为0.1)
        self.relu2 = nn.ReLU()
        self.kongdong3 = nn.Conv3d(16,16,kernel_size=(1,1,3),stride=(1,1,1),padding=(0,0,1))
        self.batch_normkongdong15 = nn.Sequential(
                                    nn.BatchNorm2d(16,  eps=0.001, momentum=0.1, affine=True)) # 动量默认值为0.1)
        self.relu3 = nn.ReLU()

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(7, 7, 7),padding=(0,3,3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv3d_features_s = nn.Sequential(
            nn.Conv3d(8, out_channels=64, kernel_size=(24, 1, 1),padding=(0,0,0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )


        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 7),padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pooling_pixal = nn.AdaptiveAvgPool1d(1)
        self.pool3d = nn.AdaptiveAvgPool2d(1)
        self.a = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, X, mask=None):  

        x = rearrange(x, 'b c h w y -> b c y h w')
        x = self.conv3d_features(x)
        x = self.conv3d_features_s(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = self.pool1(x)
        x2 = self.pool2(x)

        x = rearrange(x, 'b c h w -> b c (h w)')    
        x = self.patch_to_embedding(x)     #[b,n,dim]
        b, n, _ = x.shape       

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  
        x = torch.cat((cls_tokens, x), dim = 1) 
        x += self.pos_embedding[:, :(n + 1)] 
        x = self.dropout(x)       
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])      
        

        x1 = rearrange(x1, 'b c h w -> b c (h w)')      
        x1 = self.patch_to_embedding1(x1)     #[b,n,dim]
        b, n, _ = x1.shape        #
        cls_tokens1 = repeat(self.cls_token, '() n d -> b n d', b = b) 
        x1 = torch.cat((cls_tokens1, x1), dim = 1)
        x1 += self.pos_embedding[:, :(n + 1)] 
        x1 = self.dropout(x1)       
        x1 = self.transformer(x1, mask)
        x1 = self.to_cls_token(x1[:, 0])   


        x2 = rearrange(x2, 'b c h w -> b c (h w)')   
        x2 = self.patch_to_embedding2(x2)     #[b,n,dim]
        b, n, _ = x2.shape    
        cls_tokens2 = repeat(self.cls_token, '() n d -> b n d', b = b)  
        x2 = torch.cat((cls_tokens2, x2), dim = 1)
        x2 += self.pos_embedding[:, :(n + 1)] 
        x2 = self.dropout(x2)      
        x2 = self.transformer(x2, mask)   
        x2 = self.to_cls_token(x2[:, 0])     


        X = rearrange(X, 'b c h y -> b y (c h)')  
        identity = X
        X = self.patch_to_embedding4(X)     #[b,n,dim]
        b, n, _ = X.shape       
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)   
        X = torch.cat((cls_tokens, X), dim = 1)  
        X += self.pos_embedding1[:, :(n + 1)] 
        identity = X               

        X = self.dropout(X)       
        X = self.transformer1(X, mask)           
        X = self.to_cls_token(X[:, 0])     


        x = torch.cat((x,x1,x2),dim=-1)
        
        x = torch.cat((A*X,(1-A)*x),dim=-1)
        x = self.n(x)
        x = self.nnn(x)

        return x


if __name__ == '__main__':
    model = S2FTNet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 11, 11)   
    y = model(input)
    print(y.size())