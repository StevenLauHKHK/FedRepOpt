
# # For GhostNet model
# # FedRepOpt-GhostNet 0.5x (arch: ghost-target)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch ghost-target-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# # Fed-CSLA-Ghost 0.5x (arch: ghost-csla)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch ghost-csla-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# # Fed-RepGhost-Inf 0.5x (arch: ghost-target-norepopt)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch ghost-target-norepopt-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# # Fed-RepGhost-Tr 0.5x (arch: ghost-rep)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch ghost-rep-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For RepVGG-B1 model
# FedRepOpt-VGG-B1 (arch: RepVGG-B1-target)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch RepVGG-B1-target-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_v1/output/RepOpt-VGG-B1-hs/search/best_ckpt.pth \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10

# Fed-CSLA-VGG-B1 (arch: RepVGG-B1-csla)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch RepVGG-B1-csla-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10

# # Fed-RepVGG-B1-Inf (arch: RepVGG-B1-target-norepopt)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch RepVGG-B1-target-norepopt-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10

# # Fed-RepVGG-B1-Tr (arch: RepVGG-B1-repvgg)
python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch RepVGG-B1-repvgg-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0 \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10
