import os
import sys
import pickle
import argparse

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from PIL import Image, ImageFile
import numpy as np

from timm.models import create_model

import albumentations.pytorch
import albumentations as A
import utils
import xcit_retrieval
import gc
from tqdm import tqdm
from match_pair import compute_num_inliers, preprocess_image, get_reranking_model_input
import cv2
import numpy as np
from multiprocessing import Pool
import copy
if __name__ == '__main__':
    np.random.seed(8)
    model_backbone = create_model(
        'xcit_retrievalv2_small_12_p16',
        pretrained=False,
        num_classes=0,
    )
    model = create_model(
        'xcit_retrievalv2_small_12_p16_reduction',
        pretrained=False,
        retrieval_back_bone = model_backbone,
        num_classes=100,
    )

    checkpoint_backbone = torch.load('../model_checkpoint/xcit_small_12_p16_retrieval_5e-5-1e-7_19_tune_aug/checkpoint_model_epoch_37.pth', map_location='cpu')
    checkpoint_backbone_model = checkpoint_backbone['model']
    for k in ['head.weight']:
        print(f"Removing key {k} from pretrained backbone checkpoint")
        del checkpoint_backbone_model[k]
    model.retrieval_back_bone.load_state_dict(checkpoint_backbone_model, strict=True)
    device = torch.device("cuda")
    model.to(device)
    utils.freeze_weights(model,['retrieval_back_bone'])
    for name,param in model.named_parameters():
        if 'retrieval_back_bone' in name:
            print(name,param)
            break
    print(checkpoint_backbone_model['cls_token'])
    
    # model.eval()
    # tensor = torch.randn(4, 3, 512,512).to('cuda')
    # label = torch.tensor([5,7,8,9]).to('cuda')
    # print(model(tensor,label))
