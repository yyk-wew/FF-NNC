import torch
import torch.nn as nn

import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class ClassBlock(nn.Module):# classify the final result
    def __init__(self):
        super().__init__()
        self.c0 = nn.Linear(1024, 16)
        self.dp0 = nn.Dropout(p=0.5)

        self.c1 = nn.Linear(16, 1)
        self.dp1 = nn.Dropout(p=0.5)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        h = self.dp0(x)
        h = self.c0(h)

        h = self.leakyrelu(h)
        h = self.dp1(h)
        h = self.c1(h)
        return h


class EncodeBlock(nn.Module):
	def __init__(self, input_dim, output_dim, conv_kernel, conv_padding, pool_kernel):
		super(EncodeBlock, self).__init__()
		enc = list()

		enc += [nn.Conv2d(input_dim, output_dim, conv_kernel, padding=conv_padding, bias=False)]
		enc += [nn.LeakyReLU(inplace=True)]
		enc += [nn.BatchNorm2d(output_dim)]
		enc += [nn.MaxPool2d(kernel_size=pool_kernel)]

		self.blocks = nn.Sequential(*enc)
	
	def forward(self, x):
		return self.blocks(x)


class Meso4(nn.Module):
	def __init__(self, num_classes=1):
		super(Meso4, self).__init__()
		self.num_classes = num_classes

		enc = list()
		enc += [EncodeBlock(input_dim=3, output_dim=8, conv_kernel=3, conv_padding=1, pool_kernel=(2,2))]   # (8, 128, 128)
		enc += [EncodeBlock(input_dim=8, output_dim=8, conv_kernel=5, conv_padding=2, pool_kernel=(2,2))]   # (8, 64, 64)
		enc += [EncodeBlock(input_dim=8, output_dim=16, conv_kernel=5, conv_padding=2, pool_kernel=(2,2))]  # (16, 32, 32)
		enc_final = EncodeBlock(input_dim=16, output_dim=16, conv_kernel=5, conv_padding=2, pool_kernel=(4,4)) # (16, 8, 8)
		self.encoder = nn.Sequential(*enc)
		self.enc_final = enc_final
		
		self.fc = ClassBlock()

	def forward(self, x):
		x = self.encoder(x)
		x = self.enc_final(x)

		x = x.view(x.size()[0],-1)
		x = self.fc(x)	

		return x