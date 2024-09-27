# Hyper-parameter search on cifar100
# For RepOpt-VGG-B1 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12349 main_repopt_centralized.py \
--data-path /data_ssd/DATA/cf100 \
--arch RepOpt-VGG-B1-hs \
--batch-size 32 \
--tag search \
--opts TRAIN.EPOCHS 240 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 10 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET cf100 TRAIN.CLIP_GRAD 5.0

# For GhostNet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12339 main_repopt_centralized.py \
--data-path /data_ssd/DATA/cf100 \
--arch ghost-hs \
--batch-size 128 \
--tag search \
--opts TRAIN.EPOCHS 600 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 10 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET cf100 TRAIN.CLIP_GRAD 5.0