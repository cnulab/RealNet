
# To train the diffusion model, 4*48G GPU is required.

python -m torch.distributed.launch --nproc_per_node=4  train_diffusion.py --dataset MVTec-AD
python -m torch.distributed.launch --nproc_per_node=4  train_diffusion.py --dataset VisA
python -m torch.distributed.launch --nproc_per_node=4  train_diffusion.py --dataset MPDD
python -m torch.distributed.launch --nproc_per_node=4  train_diffusion.py --dataset BTAD

# To train the guided classifier (optional), 2*24G GPU is required.

python -m torch.distributed.launch --nproc_per_node=2  train_classifier.py --dataset MVTec-AD
python -m torch.distributed.launch --nproc_per_node=2  train_classifier.py --dataset VisA
python -m torch.distributed.launch --nproc_per_node=2  train_classifier.py --dataset MPDD
python -m torch.distributed.launch --nproc_per_node=2  train_classifier.py --dataset BTAD

# SDAS sampling, 1*24G GPU is required.

python -m torch.distributed.launch --nproc_per_node=1  sample.py --dataset MVTec-AD
python -m torch.distributed.launch --nproc_per_node=1  sample.py --dataset VisA
python -m torch.distributed.launch --nproc_per_node=1  sample.py --dataset MPDD
python -m torch.distributed.launch --nproc_per_node=1  sample.py --dataset BTAD

# To train RealNet, 1*24G GPU is required.

python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name bottle
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name cable
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name capsule
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name carpet
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name grid
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name hazelnut
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name leather
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name metal_nut
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name pill
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name screw
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name tile
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name toothbrush
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name transistor
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name wood
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name zipper

python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name candle
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name capsules
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name cashew
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name chewinggum
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name fryum
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name macaroni1
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name macaroni2
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name pcb1
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name pcb2
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name pcb3
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name pcb4
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset VisA --class_name pipe_fryum

python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MPDD --class_name bracket_black
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MPDD --class_name bracket_brown
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MPDD --class_name bracket_white
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MPDD --class_name connector
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MPDD --class_name metal_plate
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MPDD --class_name tubes

python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset BTAD --class_name 01
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset BTAD --class_name 02
python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset BTAD --class_name 03


# To evaluation RealNet, visualize and calculate PRO, 1*24G GPU is required.

python  evaluation_realnet.py --dataset MVTec-AD --class_name bottle
python  evaluation_realnet.py --dataset MVTec-AD --class_name cable
python  evaluation_realnet.py --dataset MVTec-AD --class_name capsule
python  evaluation_realnet.py --dataset MVTec-AD --class_name carpet
python  evaluation_realnet.py --dataset MVTec-AD --class_name grid
python  evaluation_realnet.py --dataset MVTec-AD --class_name hazelnut
python  evaluation_realnet.py --dataset MVTec-AD --class_name leather
python  evaluation_realnet.py --dataset MVTec-AD --class_name metal_nut
python  evaluation_realnet.py --dataset MVTec-AD --class_name pill
python  evaluation_realnet.py --dataset MVTec-AD --class_name screw
python  evaluation_realnet.py --dataset MVTec-AD --class_name tile
python  evaluation_realnet.py --dataset MVTec-AD --class_name toothbrush
python  evaluation_realnet.py --dataset MVTec-AD --class_name transistor
python  evaluation_realnet.py --dataset MVTec-AD --class_name wood
python  evaluation_realnet.py --dataset MVTec-AD --class_name zipper

python  evaluation_realnet.py --dataset VisA --class_name candle
python  evaluation_realnet.py --dataset VisA --class_name capsules
python  evaluation_realnet.py --dataset VisA --class_name cashew
python  evaluation_realnet.py --dataset VisA --class_name chewinggum
python  evaluation_realnet.py --dataset VisA --class_name fryum
python  evaluation_realnet.py --dataset VisA --class_name macaroni1
python  evaluation_realnet.py --dataset VisA --class_name macaroni2
python  evaluation_realnet.py --dataset VisA --class_name pcb1
python  evaluation_realnet.py --dataset VisA --class_name pcb2
python  evaluation_realnet.py --dataset VisA --class_name pcb3
python  evaluation_realnet.py --dataset VisA --class_name pcb4
python  evaluation_realnet.py --dataset VisA --class_name pipe_fryum

python  evaluation_realnet.py --dataset MPDD --class_name bracket_black
python  evaluation_realnet.py --dataset MPDD --class_name bracket_brown
python  evaluation_realnet.py --dataset MPDD --class_name bracket_white
python  evaluation_realnet.py --dataset MPDD --class_name connector
python  evaluation_realnet.py --dataset MPDD --class_name metal_plate
python  evaluation_realnet.py --dataset MPDD --class_name tubes

python  evaluation_realnet.py --dataset BTAD --class_name 01
python  evaluation_realnet.py --dataset BTAD --class_name 02
python  evaluation_realnet.py --dataset BTAD --class_name 03