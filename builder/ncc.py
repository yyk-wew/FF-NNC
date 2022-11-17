import torch
import torch.nn as nn
import torch.nn.functional as F

class AbsLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AbsLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        mask = torch.ones((out_features, 1))
        mask[0][0] = 0.
        self.mask = nn.Parameter(mask, requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        weight = torch.abs(self.weight) * self.mask
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init(self, bias_const=5.):
        self.weight.data.normal_(mean=0.0, std=0.01)
        self.bias.data.zero_()
        self.bias.data[0] = bias_const