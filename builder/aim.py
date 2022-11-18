import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import Transformer
import torchvision.transforms as trans
import einops


class AIM(nn.Module):
    def __init__(self):
        super(AIM, self).__init__()

        self.ATT = Transformer(dim=2048, depth=1, heads=8, dim_head=256, mlp_dim=2048)
        self.trans1 = trans.Compose([
            trans.CenterCrop(224),
            trans.Resize(256)
        ])
        self.trans2 = trans.Compose([
            trans.RandomHorizontalFlip(p=1.)
        ])
        self.att_bn = nn.BatchNorm1d(2048)

    def forward(self, x, backbone):
        """
        Note: The backbone module here should return an output feature with shape [B, C] rather than feature map.
        """
        x = torch.cat([x, self.trans1(x), self.trans2(x)], 0)

        x = backbone(x)
        
        x = einops.rearrange(x, '(b1 b2) c -> b2 b1 c', b1=3)
        x = self.ATT(x)
        x = einops.rearrange(x, 'n l c -> (n l) c')
        x = self.att_bn(x)
        x = einops.rearrange(x, '(n l) c -> n l c', l=3)
        x = F.relu(x)
        x = torch.mean(x, dim=1)
        
        return x
