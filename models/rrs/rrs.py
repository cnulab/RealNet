import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
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



class RRS(torch.nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 modes,
                 mode_numbers,
                 num_residual_layers,
                 stop_grad,
                 ):

        super(RRS, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.mode_numbers = mode_numbers
        self.modes = modes
        self.num_residual_layers=num_residual_layers
        self.stop_grad=stop_grad

        self.total_select_number=sum(self.mode_numbers)

        align_stride = min([self.instrides[block] for block in self.instrides])

        for block in self.instrides:
            self.add_module("{}_upsample".format(block),nn.UpsamplingBilinear2d(scale_factor=self.instrides[block]/align_stride))

        align_inplane = sum([self.inplanes[block] for block in self.inplanes])

        self.bn_idx = nn.BatchNorm2d(align_inplane,momentum=0.9,affine=False)

        self.decoder1 = nn.Sequential(
            ResidualStack(self.total_select_number, self.num_residual_layers),
            nn.Conv2d(self.total_select_number, 128, (3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(nn.Conv2d(128, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 8, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder3 = nn.Sequential(nn.Conv2d(8, 4, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(4, 2, (3, 3), padding=(1, 1), bias=True))


    @torch.no_grad()
    def select_ano_index(self,residual,mode, k):
        B,C,W,H = residual.size()
        residual = residual.view((B,C,W*H))
        if mode=='max':
            residual,_ = torch.max(residual,dim=-1)
        elif mode=='mean':
            residual  = torch.mean(residual, dim=-1)
        else:
            raise ValueError("mode must in [max,mean]")
        _,idxs=torch.topk(residual,dim=1,largest=True,k=k,sorted=True)
        return idxs


    def forward(self, inputs, train=False):

        residual = inputs['residual']

        if self.stop_grad:
            residual= { block :residual[block].detach() for block in residual}

        residual = torch.cat([ getattr(self,"{}_upsample".format(block))(residual[block]) for block in residual],dim=1)

        residual_idx = self.bn_idx(residual)

        B, C, H, W = residual.size()

        residual_choose = []
        for mode, mode_n in zip(self.modes, self.mode_numbers):
            idxs = self.select_ano_index(residual_idx, mode, mode_n)
            residual_choose.append(
                torch.gather(residual, dim=1, index=idxs.view((B,mode_n,1,1)).repeat(1,1,H,W)))

        residual = torch.cat(residual_choose, dim=1)

        decoded_residual = self.decoder1(residual)
        decoded_residual = self.decoder2(decoded_residual)

        upsample_size = (decoded_residual.size(-1) * 2,) * 2
        decoded_residual = F.interpolate(decoded_residual, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder3(decoded_residual)

        _, _, ht, wt = inputs['image'].size()
        logit_mask = F.interpolate(logit_mask, (ht, wt), mode='bilinear', align_corners=True)
        pred = torch.softmax(logit_mask, dim=1)
        pred = pred[:, 1, :, :].unsqueeze(1)

        return {'logit': logit_mask, "anomaly_score": pred}
