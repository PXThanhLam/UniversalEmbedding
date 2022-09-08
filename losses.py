import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import numpy as np
class RetrievalLoss(nn.Module):
    def __init__(self, multi_head_loss = False):
        super().__init__()
        self.global_criterion = nn.CrossEntropyLoss() 
        self.local_weight = 1
        self.diver_weight = 0.001
        self.multi_head_loss = multi_head_loss
        

    def forward(self, global_logits, label):
        global_loss = self.global_criterion(global_logits, label)
        return global_loss


class ReductionAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_criterion = nn.CrossEntropyLoss() 
        self.ae_weight = 12
        

    def forward(self, logit,label, glob_ori, glob_recons):
        global_loss = self.global_criterion(logit, label)
        ae_loss = torch.mean((glob_ori - glob_recons)**2)
        total_loss = global_loss + ae_loss*self.ae_weight
        return total_loss,global_loss,ae_loss
    
class ReductionAELossWoClassify(nn.Module):
    def __init__(self):
        super().__init__()        

    def forward(self, glob_ori, glob_recons, l2_loss = False):
        if l2_loss:
            ae_loss = torch.mean((glob_ori - glob_recons)**2)
            return ae_loss
        else:
            cos_loss = F.linear(glob_ori, glob_recons)
            cos_loss = torch.diagonal(cos_loss, 0)
            return 1 - torch.mean(cos_loss)

