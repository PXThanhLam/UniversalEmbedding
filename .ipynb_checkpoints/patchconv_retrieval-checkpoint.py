# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.


from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math
__all__ = ['S60', 'S120', 'B60', 'B120', 'L60', 'L120', 'S60_multi']


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Learned_Aggregation_Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class Learned_Aggregation_Layer_multi(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = (
            self.q(x[:, : self.num_classes])
            .reshape(B, self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x[:, self.num_classes:])
            .reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x[:, self.num_classes:])
            .reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_classes, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class Layer_scale_init_Block_only_token(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Learned_Aggregation_Layer,
        Mlp_block=Mlp,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, x_cls: torch.Tensor) -> torch.Tensor:
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Conv_blocks_se(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.qkv_pos = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, padding=1, stride=1, bias=True),
            nn.GELU(),
            SqueezeExcite(dim, rd_ratio=0.25),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(-1, -2)
        x = x.reshape(B, C, H, W)
        x = self.qkv_pos(x)
        x = x.reshape(B, C, N)
        x = x.transpose(-1, -2)
        return x


class Layer_scale_init_Block(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=None,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
    )


class ConvStem(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Sequential(
            conv3x3(in_chans, embed_dim // 8, 2),
            nn.GELU(),
            conv3x3(embed_dim // 8, embed_dim // 4, 2),
            nn.GELU(),
            conv3x3(embed_dim // 4, embed_dim // 2, 2),
            nn.GELU(),
            conv3x3(embed_dim // 2, embed_dim, 2),
        )

    def forward(self, x: torch.Tensor, padding_size: Optional[int] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.15):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        
        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, input, label, split = 'train'):    
        # input is not l2 normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if split == 'val':
            return cosine #* scale_factor
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor
        return logit
        
class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.2, s = 40.): # (m = 0.2, s= 40.)
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label, split = 'train'):
        embbedings = F.normalize(embbedings)
        kernel_norm = F.normalize(self.kernel)
        cos_theta = F.linear(embbedings, kernel_norm)
        if split == 'val':
            return cos_theta #* self.s
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        cos_theta_m = cos_theta_m.type(target_logit.type())
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            self.t = self.t.type(cos_theta.type())
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output

class WhiteningLayer(nn.Module):
    def __init__(self,input_emb,output_emb):
        super().__init__()
        self.linear1 = nn.Linear(input_emb,2*input_emb, bias=True)
        self.linear2 = nn.Linear(2*input_emb,input_emb, bias=True)
        self.linear3 = nn.Linear(input_emb,output_emb, bias=True)
    def forward(self,x):
        x1 = self.linear1(x)
        x2 = F.relu(self.linear2(x1)) + x
        x_reduct = self.linear3(x2)
        return x_reduct

class PatchConvnet(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        hybrid_backbone: Optional = None,
        norm_layer=nn.LayerNorm,
        global_pool: Optional[str] = None,
        block_layers=Layer_scale_init_Block,
        block_layers_token=Layer_scale_init_Block_only_token,
        Patch_layer=ConvStem,
        act_layer: nn.Module = nn.GELU,
        Attention_block=Conv_blocks_se,
        dpr_constant: bool = True,
        init_scale: float = 1e-4,
        Attention_block_token_only=Learned_Aggregation_Layer,
        Mlp_block_token_only=Mlp,
        depth_token_only: int = 1,
        mlp_ratio_clstk: float = 3.0,
        reduct = None
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        

        if not dpr_constant:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.blocks_token_only = nn.ModuleList(
            [
                block_layers_token(
                    dim=int(embed_dim),
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio_clstk,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block_token_only,
                    Mlp_block=Mlp_block_token_only,
                    init_values=init_scale,
                )
                for i in range(depth_token_only)
            ]
        )

        self.norm = norm_layer(int(embed_dim))

        self.total_len = depth_token_only + depth

        self.feature_info = [dict(num_chs=int(embed_dim), reduction=0, module='head')]
        
        self.reduct = reduct
        if reduct!=None:
            self.reduction = WhiteningLayer(embed_dim,reduct)
            embed_dim = reduct
        if num_classes !=0:
            self.head = CurricularFace(
                in_features=embed_dim,
                out_features=self.num_classes,
            )
        self.rescale: float = 0.02

        trunc_normal_(self.cls_token, std=self.rescale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.rescale)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        glob = x[:, 0]
        if self.reduct:
            glob = self.reduction(glob)
            

        return glob
    
    def get_global_feat(self,x):
        return self.forward_features(x)
    def forward(self, x, label, split = 'train'):
        B = x.shape[0]
        x = self.forward_features(x)
        x = self.head(x, label,split)
        return x


@register_model
def S60(pretrained: bool = False, **kwargs):
    model = PatchConvnet(
        patch_size=16,
        embed_dim=384,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        depth_token_only=1,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


@register_model
def S120(pretrained: bool = False, **kwargs):
    model = PatchConvnet(
        patch_size=16,
        embed_dim=384,
        depth=120,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


@register_model
def B60(pretrained: bool = False, **kwargs):
    model = PatchConvnet(
        patch_size=16,
        embed_dim=768,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        **kwargs
    )

    return model


@register_model
def B120(pretrained: bool = False, **kwargs):
    model = PatchConvnet(
        patch_size=16,
        embed_dim=768,
        depth=120,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        **kwargs
    )

    return model


@register_model
def L60(pretrained: bool = False, **kwargs):
    model = PatchConvnet(
        patch_size=16,
        embed_dim=1024,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


@register_model
def L120(pretrained: bool = False, **kwargs):
    model = PatchConvnet(
        patch_size=16,
        embed_dim=1024,
        depth=120,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


