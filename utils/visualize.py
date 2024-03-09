
import os
import matplotlib
import numpy as np
from PIL import Image
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam.utils.image import show_cam_on_image


def export_sdas_images(config, fileinfos, gen_images, recon_images,epoch):

    export_root=os.path.join(config.vis_path,str(epoch))
    os.makedirs(export_root,exist_ok=True)

    for info,gen_image,recon_image in tqdm(zip(fileinfos,gen_images,recon_images)):

        clsname = info['clsname']
        clsname_root= os.path.join(export_root,clsname)
        os.makedirs(clsname_root,exist_ok=True)

        info = info['filename']
        info, filename = os.path.split(info)
        _, subname = os.path.split(info)
        subname_root=os.path.join(clsname_root,subname)
        os.makedirs(subname_root,exist_ok=True)

        ori_image=np.clip((gen_image.transpose(1,2,0) +1)* 127.5, a_min=0, a_max=255)
        recon_image= np.clip((recon_image.transpose(1,2,0) +1)* 127.5, a_min=0, a_max=255)

        merge_image = np.concatenate([ori_image,recon_image], axis=0).astype(np.uint8)

        Image.fromarray(merge_image).save(os.path.join(subname_root,filename))



def export_segment_images(config, test_img_paths, gts, scores, threshold,class_name):
    image_dirs = os.path.join(config.vis_path, class_name)

    print('Exporting images...')
    os.makedirs(image_dirs, exist_ok=True)
    num = len(test_img_paths)
    scores_norm = 1.0/scores.max()

    for i in tqdm(range(num)):
        img = np.array(Image.open(os.path.join(config.dataset.image_reader.kwargs.image_dir,
                                               test_img_paths[i])).convert("RGB").resize(config.dataset.input_size))

        filedir,filname=os.path.split(test_img_paths[i])
        filedir, subname = os.path.split(filedir)

        save_path=os.path.join(image_dirs,subname+"_"+filname)

        gt_mask = gts[i].astype(np.uint8)[...,None].repeat(3,-1)

        score_mask = np.zeros_like(scores[i])
        score_mask[scores[i] >  threshold] = 1.0
        score_mask = (255.0*score_mask).astype(np.uint8)

        score_map = scores[i] * scores_norm

        heat = show_cam_on_image(img / 255, score_map, use_rgb=True)

        score_img = mark_boundaries(heat, score_mask, color=(1, 0, 0), mode='thick')

        merge=np.concatenate([img,255*gt_mask,255*score_img],axis=1).astype(np.uint8)
        Image.fromarray(merge).save(save_path)
