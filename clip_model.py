import os
import clip
from clip.clip import _download, _MODELS
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import math
import numpy as np



class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   
    
class ArcFaceAdaptiveMargin(nn.modules.Module):
    def __init__(self,in_features, out_features, s= 45.0):
        super().__init__()
        self.s = s
        
        num_per_class = np.zeros((124379,))
        with open('../split_path/train_split_reduce.txt', 'r') as t_f:
            for data in t_f.readlines():
                im_dir, cont_id = data.strip().split("---")
                num_per_class[int(cont_id)] += 1
        t_f.close()
        num_per_class = num_per_class + 1
        margin_per_class = num_per_class ** (-0.25)
        margin_per_class = (margin_per_class - margin_per_class.min()) / (margin_per_class.max() - margin_per_class.min()) * 0.45 + 0.07

        self.margins = margin_per_class
        self.out_dim = out_features
        self.kernel = ArcMarginProduct_subcenter(in_features, out_features)
            
    def forward(self, embbedings, labels, split = 'train'):
        logits = self.kernel(embbedings)
        if split == 'val':
            return logits * self.s
        ms = []
        try:
            ms = self.margins[labels.cpu().numpy()]
        except:
            ms = np.array([0.3] * len(labels.cpu().numpy()))
            print('Error', labels.cpu().numpy())
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output     

    
class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.3, s = 45.): # (m = 0.2, s= 40.)
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


class ClipvitlModel(nn.Module):
    def __init__(self, load_pretrained_clip = True, n_class = 1000):
        super().__init__()
        clip_model, _ = clip.load('ViT-L/14@336px',jit=False,device=torch.device('cpu'))
        del(clip_model.transformer)
        self.vit_model = clip_model.visual
        embed_dim = 768
        if n_class !=0:
            self.head = CurricularFace(
                in_features=embed_dim,
                out_features=n_class,
            )
    def get_global_feat(self,x):
        return self.forward_features(x)
    def forward_features(self, x):
        x = self.vit_model(x)
        return x
    def forward(self, x, label, split = 'train'):
        x = self.forward_features(x)
        x = self.head(x, label,split)
        return x
# python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --num_workers 3 --batch-size 96 --input-size 336 --from_timm False --output_dir model_checkpoint/vitl_clip_curface_datasetv3_keep_22_23 --warmup-lr 1.3e-4 --lr 7e-5 --warmup-epochs 1 --resume True --eval


class ClipvitlModel2(nn.Module):
    def __init__(self, load_pretrained_clip = True, n_class = 1000):
        super().__init__()
        clip_model, _ = clip.load('ViT-L/14@336px',jit=False,device=torch.device('cpu'))
        del(clip_model.transformer)
        self.vit_model = clip_model.visual
        dim = 768
        h_dim = 1024
        embed_dim = 1024
        self.projector = torch.nn.Sequential(
                nn.Linear(dim,h_dim),
                nn.GELU(),
                nn.Linear(h_dim,embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim,embed_dim),
            )
        
        if n_class !=0:
            self.head = ArcFaceAdaptiveMargin(
                in_features=embed_dim,
                out_features=n_class,
            )
    def get_global_feat(self,x):
        return self.forward_features(x)
    def forward_features(self, x):
        x = self.vit_model(x)
        x = self.projector(x)
        return x
    def forward(self, x, label, split = 'train'):
        x = self.forward_features(x)
        x = self.head(x, label,split)
        return x

class ClipvitlModel3(nn.Module):
    def __init__(self, load_pretrained_clip = True, n_class = 1000):
        super().__init__()
        clip_model, _ = clip.load('ViT-L/14@336px',jit=False,device=torch.device('cpu'))
        del(clip_model.transformer)
        self.vit_model = clip_model.visual
        dim = 768
        h_dim = 1024
        embed_dim = 1344
        self.projector = torch.nn.Sequential(
                nn.Linear(dim,h_dim),
                nn.GELU(),
                nn.Linear(h_dim,h_dim),
                nn.GELU(),
                nn.Linear(h_dim,h_dim),
                nn.GELU(),
                nn.Linear(h_dim,embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim,embed_dim),
            )
        
        if n_class !=0:
            self.head = ArcFaceAdaptiveMargin(
                in_features=embed_dim,
                out_features=n_class,
            )
    def get_global_feat(self,x):
        return self.forward_features(x)
    def forward_features(self, x):
        x = self.vit_model(x)
        x = self.projector(x)
        return x
    def forward(self, x, label, split = 'train'):
        x = self.forward_features(x)
        x = self.head(x, label,split)
        return x



class ClipvitlModelReduction(nn.Module):
    def __init__(self, pretrained_backbone, reduct_dim = 64):
        super().__init__()
        self.retrieval_back_bone = pretrained_backbone
        dim = 768
        self.encoder = torch.nn.Sequential(
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,reduct_dim)
            )
        self.decoder = nn.Linear(reduct_dim,dim)
        embed_dim = 768
    def get_global_feat(self,x):
        return self.forward_features(x)[2]
    def forward_features(self, x):
        self.retrieval_back_bone.eval()
        x = self.retrieval_back_bone.get_global_feat(x)
        original_glob = x.detach()
        x_reduct = self.encoder(original_glob)
        x_recons = self.decoder(x_reduct)
        return original_glob,x_recons,x_reduct
    def forward(self, x, label, split = 'train'):
        glob_ori,glob_recons, glob_reduct  = self.forward_features(x)
        return glob_ori,glob_recons
    
class ClipvitlModelReduction2(nn.Module):
    def __init__(self, pretrained_backbone, reduct_dim = 64):
        super().__init__()
        self.retrieval_back_bone = pretrained_backbone
        dim = 1024
        self.encoder = torch.nn.Sequential(
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,reduct_dim)
            )
        self.decoder = torch.nn.Sequential(
                nn.Linear(reduct_dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
            )
    def get_global_feat(self,x):
        return self.forward_features(x)[2]
    def forward_features(self, x):
        self.retrieval_back_bone.eval()
        x = self.retrieval_back_bone.get_global_feat(x)
        original_glob = x.detach()
        x_reduct = self.encoder(original_glob)
        x_recons = self.decoder(x_reduct)
        return original_glob,x_recons,x_reduct
    def forward(self, x, label, split = 'train'):
        glob_ori,glob_recons, glob_reduct  = self.forward_features(x)
        return glob_ori,glob_recons


class ClipvitlModelReduction3(nn.Module):
    def __init__(self, pretrained_backbone, reduct_dim = 64):
        super().__init__()
        self.retrieval_back_bone = pretrained_backbone
        dim = 1024
        self.encoder = torch.nn.Sequential(
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,reduct_dim)
            )
        self.decoder = torch.nn.Sequential(
                nn.Linear(reduct_dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
                nn.GELU(),
                nn.Linear(dim,dim),
            )
    def get_global_feat(self,x):
        return self.forward_features(x)[2]
    def forward_features(self, x):
        self.retrieval_back_bone.eval()
        x = self.retrieval_back_bone.get_global_feat(x)
        x = F.normalize(x)
        original_glob = x.detach()
        x_reduct = self.encoder(original_glob)
        x_reduct = F.normalize(x_reduct)
        x_recons = self.decoder(x_reduct)
        x_recons = F.normalize(x_recons)
        return original_glob,x_recons,x_reduct
    def forward(self, x, label, split = 'train'):
        glob_ori,glob_recons, glob_reduct  = self.forward_features(x)
        return glob_ori,glob_recons


