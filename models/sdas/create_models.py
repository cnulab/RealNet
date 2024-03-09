import torch
from .unets import EncoderUNetModel,UNetModel
from utils.misc_helper import load_pertrain_weights


def create_diffusion_unet(
        image_size,
        classifier_width,
        classifier_depth,
        learn_sigma,
        attention_resolutions,
        dropout,
        channel_mult,
        class_number,
        num_heads,
        num_heads_upsample,
        num_head_channels,
        use_scale_shift_norm,
        resblock_updown,
        use_fp16,
        pertrain_path,
):
    unet = UNetModel(       image_size=image_size,
                            in_channels=3,
                            model_channels=classifier_width,
                            out_channels=(6 if learn_sigma else 3),
                            num_res_blocks=classifier_depth,
                            attention_resolutions=attention_resolutions,
                            dropout=dropout,
                            channel_mult=channel_mult,
                            num_classes=class_number,
                            num_heads=num_heads,
                            use_fp16=use_fp16,
                            num_heads_upsample=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_scale_shift_norm=use_scale_shift_norm,
                            resblock_updown=resblock_updown
                        )

    if pertrain_path is not None:
        missing_keys, unexpected_keys = load_pertrain_weights(pertrain_path,unet)
        # print(missing_keys, unexpected_keys)
    return unet



def create_classifier_unet(
        image_size,
        classifier_width,
        class_number,
        classifier_depth,
        classifier_attention_resolutions,
        channel_mult,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        pertrain_path=None,
    ):

    classifier=EncoderUNetModel(image_size=image_size,
                                in_channels=3,
                                model_channels=classifier_width,
                                out_channels=class_number,
                                num_res_blocks=classifier_depth,
                                attention_resolutions=classifier_attention_resolutions,
                                dropout=0,
                                channel_mult=channel_mult,
                                conv_resample=True,
                                num_heads=1,
                                num_heads_upsample=-1,
                                num_head_channels=64,
                                use_scale_shift_norm=classifier_use_scale_shift_norm,
                                resblock_updown=classifier_resblock_updown,
                                pool=classifier_pool,
                                )

    if pertrain_path is not None:
        missing_keys, unexpected_keys= load_pertrain_weights(pertrain_path,classifier)
        # print(missing_keys, unexpected_keys)

    return classifier

