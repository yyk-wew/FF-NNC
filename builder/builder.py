import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .mesonet import Meso4
from torchvision.models.resnet import resnet50
from .xception import xception
from .aim import AIMWrapper
from .ncc import AbsLinear

import torchvision.transforms as trans


class Trainer(nn.Module):
    def __init__(self, model_name='resnet',use_mc=False, use_ncc=False, use_aim=False):
        """
        model_name: name of backbone module ('resnet', 'xception', 'mesonet')
        use_mc: whether to use multi-class supervision 
        use_ncc: whether to use non-negative linear classifier
        use_aim: whether to use augmented integration module
        """
        super(Trainer, self).__init__()

        # init backbone
        self._get_backbone(model_name=model_name)

        # init AIM
        self.use_aim = use_aim
        if self.use_aim:
            self.AIM = AIMWrapper(self.backbone)

        # init classifier
        self.use_mc = use_mc
        out_dim = 5 if use_mc else 2

        self.use_ncc = use_ncc
        if use_ncc:
            self.fc = AbsLinear(2048, out_dim)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()
            self.fc.bias.data[0] = 5.
        else:
            self.fc = nn.Linear(2048, out_dim)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()


    def _get_backbone(self, model_name):

        if model_name == 'mesonet':
            self.backbone = Meso4()
        elif model_name == 'resnet':
            self.backbone = resnet50(pretrained=False)
        elif model_name == 'xception':
            self.backbone = xception(pretrained=False)
        else:
            raise RuntimeError("Unsupported backbone.")

        self.backbone.fc = nn.Identity()


    def forward(self, x):
        if self.use_aim:
            x = self.AIM(x)
        else:
            x = self.backbone(x)
        
        score = self.fc(x)
   
        if self.training:
            return score
        else:
            return self.trans_to_eval(score)

    def trans_to_eval(self, x):
        # Transform N-class probability to binary form
        x = 1. - torch.softmax(x, dim=1)[:,0]
        return x

    def convert_label_to_binary(self, label_5):
        # Transform N-class label to binary form
        l = torch.where(label_5 >= 1, 1, label_5)
        return l

    def load_pretrained_backbone(self, path):
        state_dict = torch.load(path)

        # A trick to handle imagenet checkpoints and ours
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']

        # A trick to handle xception checkpoints
        if 'xception-b5690688' in path:
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}

        # Remove fc param
        fc_key_list = []
        for key, item in state_dict.items():
            if 'fc' in key:
                fc_key_list.append(key)
        for key in fc_key_list:
            del state_dict[key]

        # load
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        return msg


    def save_ckpt(self, ckpt_path):
        state_dict = {'model':self.state_dict()}
        torch.save(state_dict, ckpt_path)
