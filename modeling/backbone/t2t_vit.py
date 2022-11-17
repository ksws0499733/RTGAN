# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding



class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=(224,224), tokens_type='performer', in_chans=3, 
                        embed_dim=768, token_dim=64):
        super().__init__()

        self.token_dim=token_dim
        self.embed_dim=embed_dim
        print('    tokens_type:',tokens_type)
        if tokens_type == 'transformer':
            
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            # print('    adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            # print('    adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size[0] // (4 * 2 * 2)) * (img_size[1] // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        b,_,h,w = x.shape
        C1 = self.soft_split0(x).transpose(1, 2).contiguous()
        C1 = self.attention1(C1)
        # _, _, c = C1.shape
        C1 = C1.transpose(1,2).contiguous().reshape(b, self.token_dim, h//4, w//4)

        C2 = self.soft_split1(C1).transpose(1, 2).contiguous()
        C2 = self.attention2(C2)
        # _, _, c = C2.shape        
        C2 = C2.transpose(1, 2).contiguous().reshape(b, self.token_dim, h//8, w//8)

        C3 = self.soft_split2(C2).transpose(1, 2).contiguous()
        xout = self.project(C3)

        # _, _, c = xout.shape        
        C3 = xout.transpose(1, 2).contiguous().reshape(b, self.embed_dim, h//16, w//16) 

        return xout,[C1,C2,C3]

class T2T_ViT(nn.Module):
    def __init__(self, img_size=(224,224), tokens_type='performer', 
                 in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, 
                tokens_type=tokens_type, 
                in_chans=in_chans, 
                embed_dim=embed_dim, 
                token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches
        print('num_patches',num_patches)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=num_patches, 
                                       d_hid=embed_dim), 
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                        Block(
                            dim=embed_dim, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate, 
                            drop_path=dpr[i], 
                            norm_layer=norm_layer)
                    for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        b,_,h,w = x.shape
        C4,side = self.tokens_to_token(x)

        # _,wh,_ = C4.shape
        # cls_tokens = self.cls_token.expand(b, wh, -1)
        # C4 = torch.cat((cls_tokens, C4), dim=1)
        C4 = C4 + self.pos_embed
        C4 = self.pos_drop(C4)

        for blk in self.blocks:
            C4 = blk(C4)
        # C4 = C4[:,0:wh]
        C4 = self.norm(C4)
        # _, _, c = C4.shape        
        C4 = C4.transpose(1, 2).contiguous().reshape(b, self.embed_dim, h//16, w//16) 

        C1,C2,C3 = side

        return C1,C2,C3,C4

    def forward(self, x):
        return self.forward_features(x)
