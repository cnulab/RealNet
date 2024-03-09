import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels//2,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels//2,
                      out_channels=in_channels,
                      kernel_size=1, stride=1,bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class SimpleUnet(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(SimpleUnet, self).__init__()
        norm_layer = nn.InstanceNorm2d
        # norm_layer = nn.BatchNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels , in_channels, kernel_size=3, padding=1),
            norm_layer(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU()
        )

        self.mp1 = nn.Sequential(nn.AvgPool2d(2))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            norm_layer(in_channels * 4),
            nn.ReLU())

        self.mp2 = nn.Sequential(nn.AvgPool2d(2))

        self._residual_stack = ResidualStack(in_channels*4 ,num_residual_layers)

        self.upblock1 = nn.ConvTranspose2d(in_channels=in_channels * 8,
                                           out_channels=in_channels * 2,
                                           kernel_size=4,
                                           stride=2, padding=1)

        self.upblock2 = nn.ConvTranspose2d(in_channels=in_channels*4,
                                            out_channels=in_channels,
                                            kernel_size=4,
                                            stride=2, padding=1)

    def forward(self, inputs):
        x = self.block1(inputs)
        b1 =self.mp1(x)
        x = self.block2(b1)
        b2 = self.mp2(x)
        x = self._residual_stack(b2)
        x = self.upblock1(torch.cat([x,b2],dim=1))
        x = F.relu(x)
        x = self.upblock2(torch.cat([x,b1],dim=1))
        return x



class SimpleReconstructionLayer(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 num_residual_layers,
                 ):

        super(SimpleReconstructionLayer, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.num_residual_layers=num_residual_layers

        for block_name in self.inplanes:
            module= SimpleUnet(in_channels=self.inplanes[block_name],num_residual_layers=self.num_residual_layers)
            self.add_module('{}_recon'.format(block_name),module)

    def forward(self, inputs,train=False):
        block_feats=inputs['block_feats']
        recon_feats={ block_name:getattr(self,'{}_recon'.format(block_name))(block_feats[block_name]) for block_name in block_feats}
        residual={ block_name: (block_feats[block_name]- recon_feats[block_name] )**2 for block_name in block_feats}
        return {'feats_recon':recon_feats,'residual':residual}

    def get_outplanes(self):
        return self.inplanes

    def get_outstrides(self):
        return self.instrides