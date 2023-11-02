import torch
import torch.nn as nn
import numpy as np


class LogitScaleNetwork(nn.Module):
    def __init__(self, init_scale=1 / 0.07):
        super(LogitScaleNetwork, self).__init__()
        # from openclip/model.py. The initial scale is ln(100)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(init_scale))

    def forward(self, x=None):  # add x to make it compatible with DDP
        return self.logit_scale.exp()

"""
class LogitScaleNetwork(nn.Module):
    def __init__(self, init_scale=1 / 0.07):
        super(LogitScaleNetwork, self).__init__()
        # from openclip/model.py. The initial scale is ln(100)
        self.image_logit_scale = nn.Parameter(torch.ones([]) * np.log(init_scale))
        self.text_logit_scale = nn.Parameter(torch.ones([]) * np.log(init_scale))
        
    def forward(self, x=None): # add x to make it compatible with DDP
        return self.image_logit_scale.exp(), self.text_logit_scale.exp()
"""