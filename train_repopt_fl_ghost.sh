# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2


# # # ==========================================
# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED0-1-v2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED1-2-v2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# # # ==========================================
# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0



# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# =======================================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-cifa100-lr-0.01-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-cifa100-lr-0.01-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-cifa100-lr-0.01-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# # =======================================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-csla-cifa100-lr-0.01-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-csla-cifa100-lr-0.01-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1
# # =======================================================


# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2


# # =======================================================
# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-csla-cifa100-lr-0.01-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# # =======================================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-norepopt-cifa100-lr-0.01-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-norepopt-cifa100-lr-0.01-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-norepopt-cifa100-lr-0.01-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# # =======================================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-cifa100-lr-0.01-1LE-240Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-cifa100-lr-0.01-1LE-240Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-cifa100-lr-0.01-1LE-240Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2

# # =======================================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-csla-cifa100-lr-0.01-1LE-240Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-csla-cifa100-lr-0.01-1LE-240Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-csla-cifa100-lr-0.01-1LE-240Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2

# # =======================================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-norepopt-cifa100-lr-0.01-1LE-240Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-norepopt-cifa100-lr-0.01-1LE-240Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-target-norepopt-cifa100-lr-0.01-1LE-240Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET cifa100 DATA.IMG_SIZE 32 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2





# # # ==========================================
# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED2-3-v2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2

# # # ==========================================


# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2


# # # ==========================================
# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2

# # # # ==========================================
# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-original-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-original-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/cifa100_images \
# --arch ghost-original-tinyImageNet-lr-0.01-1LE-240Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 240 \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 3


# # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2


# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-norepopt-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2


# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2


# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2


# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2


# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-csla-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-IID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_1e5_clients_10 SEED 2


# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-mom0.9-10LE-24Rds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 24 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 10 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.9 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2

# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-target-tinyImageNet-lr-0.01-1LE-1000Rds-100clients-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 10 \
# --scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs/reproduce/best_ckpt.pth \
# --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 10 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 10 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 1


# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 10 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 2

# # # # # ==========================================
# python src_fl/main.py \
# --data-path /data_ssd/DATA/femnist_subset \
# --arch ghost-rep-femnist-lr-0.01-5LE-500Rds-268clients-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 500 \
# --pool_size 268 \
# --num_clients_per_round 27 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET femnist DATA.IMG_SIZE 28 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_non_iid_268 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/femnist_subset \
# --arch ghost-rep-femnist-lr-0.01-5LE-500Rds-268clients-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 500 \
# --pool_size 268 \
# --num_clients_per_round 27 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET femnist DATA.IMG_SIZE 28 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_non_iid_268 SEED 1

# python src_fl/main.py \
# --data-path /data_ssd/DATA/femnist_subset \
# --arch ghost-rep-femnist-lr-0.01-5LE-500Rds-268clients-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 500 \
# --pool_size 268 \
# --num_clients_per_round 27 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET femnist DATA.IMG_SIZE 28 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_non_iid_268 SEED 2


# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-5clientsperrounds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 5 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-5clientsperrounds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 5 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 1


# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-5clientsperrounds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 5 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 2

# # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-50clientsperrounds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 50 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-50clientsperrounds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 50 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 1


# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-50clientsperrounds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 50 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 2

# # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-70clientsperrounds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 70 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-70clientsperrounds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 70 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 1


# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-70clientsperrounds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 70 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 2

# # # # # ==========================================

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-90clientsperrounds-NIID-SEED0-1 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 90 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 0

# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-90clientsperrounds-NIID-SEED1-2 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 90 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 1


# python src_fl/main.py \
# --data-path /data_ssd/DATA/tiny-imagenet-200 \
# --arch ghost-rep-tinyImageNet-lr-0.01-5LE-1000Rds-100clients-90clientsperrounds-NIID-SEED2-3 \
# --batch-size 32 \
# --tag experiment \
# --rounds 1000 \
# --pool_size 100 \
# --num_clients_per_round 90 \
# --opts TRAIN.EPOCHS 5 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
# DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_100 SEED 2

# # # # # ==========================================

python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch ghost-target-tinyImageNet-iNaturalSearch-lr-0.01-1LE-240Rds-NIID-SEED0-1 \
--batch-size 32 \
--tag experiment \
--rounds 240 \
--scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs-iNaturalist/reproduce/best_ckpt.pth \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0

python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch ghost-target-tinyImageNet-iNaturalSearch-lr-0.01-1LE-240Rds-NIID-SEED1-2 \
--batch-size 32 \
--tag experiment \
--rounds 240 \
--scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs-iNaturalist/reproduce/best_ckpt.pth \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 1

python src_fl/main.py \
--data-path /data_ssd/DATA/tiny-imagenet-200 \
--arch ghost-target-tinyImageNet-iNaturalSearch-lr-0.01-1LE-240Rds-NIID-SEED2-3 \
--batch-size 32 \
--tag experiment \
--rounds 240 \
--scales-path /data1/steven/Rep_Fred/RepOptimizers_cent_ghost/output/ghost-hs-iNaturalist/reproduce/best_ckpt.pth \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 2