
### Download checkpoints:
- **MVTec-AD: [[Diffusion]](https://drive.google.com/file/d/1cl2w5eCFrmbOEWlcqPakignI4CHZdm_d/view?usp=drive_link), [[Guided Classifier]](https://drive.google.com/file/d/1-geYTTmeDD9yZzEtstbgjwlWFfgNhld8/view?usp=drive_link)**  
- **MPDD: [[Diffusion]](https://drive.google.com/file/d/1GYUdxObhgu-kWIwBf6gumsMOY3IZDF6o/view?usp=drive_link), [[Guided Classifier]](https://drive.google.com/file/d/1pLEOk4D5o80Yzq7RDeSSBv2HaF76Fxf5/view?usp=drive_link)**  
- **BTAD: [[Diffusion]](https://drive.google.com/file/d/1IYktXaXIOCv3otIVTmvj2Ck5DOeLHXiN/view?usp=drive_link), [[Guided Classifier]](https://drive.google.com/file/d/1ASS70U72VOVcAqaN4AK-EZlgEGqfj1p3/view?usp=drive_link)**  
- **VisA: [[Diffusion]](https://drive.google.com/file/d/1FzgW5xRz-TtPBkbMBbSoAJDq5gkO6Yd_/view?usp=drive_link), [[Guided Classifier]](https://drive.google.com/file/d/15bdOwBdO_bd74p2rcIKt9pTMDzxgjoJW/view?usp=drive_link)**  


### The complete directory structure is as follows:
```
    |--experiments                         
        |--MVTec-AD           
            |--diffusion_checkpoints
                |--mvtec_diffusion_ckpt_epoch_240.pt
            |--classifier_checkpoints
                |--mvtec_classifier_ckpt_epoch_428.pt
            |--diffusion.yaml    
            |--classifier.yaml
            |--sample.yaml    
            |--realnet.yaml
        |--MPDD           
            |--diffusion_checkpoints
                |--mpdd_diffusion_ckpt_epoch_825.pt
            |--classifier_checkpoints
                |--mpdd_classifier_ckpt_epoch_805.pt
            |--diffusion.yaml    
            |--classifier.yaml
            |--sample.yaml    
            |--realnet.yaml
        |--BTAD           
            |--diffusion_checkpoints
                |--btad_diffusion_ckpt_epoch_375.pt
            |--classifier_checkpoints
                |--btad_classifier_ckpt_epoch_685.pt
            |--diffusion.yaml    
            |--classifier.yaml
            |--sample.yaml    
            |--realnet.yaml
        |--VisA      
            |--diffusion_checkpoints
                |--visa_diffusion_ckpt_epoch_120.pt
            |--classifier_checkpoints
                |--visa_classifier_ckpt_epoch_188.pt
            |--diffusion.yaml    
            |--classifier.yaml
            |--sample.yaml    
            |--realnet.yaml
```
