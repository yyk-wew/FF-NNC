from itertools import count
from thop import profile as thop_profile
from thop import clever_format
from thop.vision.basic_hooks import count_linear
import torch
import torch.nn as nn
from builder.builder import Trainer
from builder.ncc import AbsLinear


# Init
dummy_input = torch.randn(1, 3, 256, 256)
baseline = Trainer(model_name='xception')
ours = Trainer(model_name='xception', use_mc=True, use_aim=True)

# Computational Cost
def count_macs_of_ncc(m, x, y):
    # mask + linear
    m.total_ops += torch.DoubleTensor([int(m.out_features * m.in_features + m.in_features * y.numel())])

custom_ops = {AbsLinear: count_macs_of_ncc}
macs, params = clever_format(thop_profile(model=baseline, inputs=(dummy_input, ), custom_ops=custom_ops, verbose=False), "%.3f")
print("Basline, MACs:{}, Params:{}".format(macs, params))
macs, params = clever_format(thop_profile(model=ours, inputs=(dummy_input, ), custom_ops=custom_ops, verbose=False), "%.3f")
print("Baseline + NCC + AIM, MACs:{}, Params:{}".format(macs, params))
