import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from models.sdas.model_utils import (
        normalization,
        Downsample,
        zero_module,
        AttentionBlock
)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.ConvTranspose2d(in_channels=channels,
                                           out_channels=self.out_channels,
                                           kernel_size=4,
                                           stride=2, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv:
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x


class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True)
            self.x_upd = Upsample(channels, True)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                 channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d( channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        return self.skip_connection(x) + h



class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        channel_mult,
        attention_mult,
        num_heads = 4,
        num_heads_upsample=-1,
        num_head_channels = 64,
    ):

        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.channel_mult = channel_mult

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, ch, 3, padding=1)]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        out_channels=int(mult * model_channels),
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_mult:
                    layers.append(
                                AttentionBlock(
                                    ch,
                                    num_heads=num_heads,
                                    num_head_channels=num_head_channels,
                            )
                    )

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            down=True,
                        )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch


        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        out_channels=int(model_channels * mult),
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_mult:
                    layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,##
                                num_head_channels=num_head_channels,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            up=True,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )


    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
        return self.out(h)


class ReconstructionLayer(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 num_res_blocks,
                 hide_channels_ratio,
                 channel_mult,
                 attention_mult
                 ):

        super(ReconstructionLayer, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.num_res_blocks=num_res_blocks
        self.attention_mult=attention_mult

        for block_name in self.inplanes:
            module= UNetModel(
                in_channels=self.inplanes[block_name],
                out_channels=self.inplanes[block_name],
                model_channels=int(hide_channels_ratio*self.inplanes[block_name]),
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                attention_mult=attention_mult
            )
            self.add_module('{}_recon'.format(block_name),module)


    def forward(self, inputs,train=False):
        block_feats = inputs['block_feats']
        recon_feats = { block_name:getattr(self,'{}_recon'.format(block_name))(block_feats[block_name]) for block_name in block_feats}
        residual={ block_name: (block_feats[block_name] - recon_feats[block_name] )**2 for block_name in block_feats}
        return {'feats_recon':recon_feats,'residual':residual}


    def get_outplanes(self):
        return self.inplanes

    def get_outstrides(self):
        return self.instrides
