# FedRepOpt (ACCV 2024)

This repository implements the model proposed in the ACCV 2024 paper:

Kin Wai Lau, Yasar Abbas Ur Rehman, Pedro Porto Buarque de Gusmão, Lai-Man Po, Lan Ma, Yuyang Xie, **FedRepOpt: Gradient Re-parameterized Optimizers in Federated Learning** [[arXiv paper]](https://arxiv.org/abs/2409.15898)

The implementation code is based on the **Re-parameterizing Your Optimizers rather than Architectures**, ICLR, 2023. For more information, please refer to the [link](https://github.com/DingXiaoH/RepOptimizers).

## Citing

When using this code, kindly reference:

```
@misc{lau2024fedrepoptgradientreparameterizedoptimizers,
      title={FedRepOpt: Gradient Re-parameterized Optimizers in Federated Learning}, 
      author={Kin Wai Lau and Yasar Abbas Ur Rehman and Pedro Porto Buarque de Gusmão and Lai-Man Po and Lan Ma and Yuyang Xie},
      year={2024},
      eprint={2409.15898},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.15898}, 
}
```

## Pretrained models
You can download our **hyperparameter searched models** on **CIFAR100** as follow:
- GhostNet-Tr 0.5x (arch: ghost-hs) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/El031zqYQJlNhENVTTaB0wQBWEzELhH6RxUkxdPx_j46Bg?e=Opp7Jz)
- RepVGG-B1 (arch:RepOpt-VGG-B1-hs) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EmyxljNO_VhFvYpeCP5yLZ8BlQosG82WPFqT_WAM_-8erQ?e=aZEonT)


You can download our pretrained models on **Tiny ImagenNet** as follow:
Pretrained on 1 local epoch and 240 rounds with **cross-silo NIID** setting
- Fed-RepGhost-Tr 0.5x (arch: ghost-rep) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EnS5vyOHtWxCq_aTKh5zB-IBuwORMB_UZSxi8wL88GelsA?e=uJPlfU)
- Fed-RepGhost-Inf 0.5x (arch: ghost-target-norepopt) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EjJ4eLKNJLxBkRWhvkuBX9QB2x6aOrX0yjMwV7FDvkU81Q?e=gTypBU)
- Fed-CSLA-Ghost 0.5x (arch: ghost-csla) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EpQimq4m-LJAk6nsZLkajW8BQ4lLny0KDDWw0ugMOv0low?e=BHyrgD)
- FedRepOpt-GhostNet 0.5x (arch: ghost-target) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Em8zwl41N1RMqhpu-YJgTAUBqwMx3mT2nvPBE4-FH_eNYg?e=9C0kcC)

- Fed-RepVGG-B1-Tr (arch: RepVGG-B1-repvgg) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EvV9sp_jDqpNj4v_N7Myea0BPX2dDphS50MsmzRT4zzSWA?e=kYeteR)
- Fed-RepVGG-B1-Inf (arch: RepVGG-B1-target-norepopt) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Ej8YlHfu9SBKmjY1Dkb838wBlXQD_PGvSNBCyOMBSrNnRQ?e=fK068t)
- Fed-CSLA-VGG-B1 (arch: RepVGG-B1-csla) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/ErLWqdjF7rlLhKFubTnooA0B_KGNLe2gL1nesDgexlkzmQ?e=6EhGI8)
- FedRepOpt-VGG-B1 (arch: RepVGG-B1-target) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Eq2WDh2W2GlAlh82BS0ViSsB2gdANrcyiNglugFt5DkHiw?e=lqR4Xo)

Pretrained on 5 local epoch and 1000 rounds with **cross-device NIID** setting
- Fed-RepGhost-Tr 0.5x (arch: ghost-rep) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EjgsygZWzolElYlDY_REUB0BBSNKWNc0MHypZO_ZfX3v0A?e=b0CyUo)
- Fed-RepGhost-Inf 0.5x (arch: ghost-target-norepopt) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Ej5K5OK2vt5InYHq_z846dUB6Qcf1kvuRPSrsSX5HmwzzQ?e=gDa2dX)
- Fed-CSLA-Ghost 0.5x (arch: ghost-csla) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EiaHbW6rdDRKiz3vlfiUmQEBEB4Z8eHXdjnZdl2awOwY1A?e=YEW107)
- FedRepOpt-GhostNet 0.5x (arch: ghost-target) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Eg0Ay1s5uwFLgp07aVKsm-kBa3VLJFXwNpmHubfVh4TZNg?e=4ORCMS)

- Fed-RepVGG-B1-Tr (arch: RepVGG-B1-repvgg) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EqrTlb5whpRHgctSpqQViOAB1xEHSM1uF7Zs01DtPku-ug?e=ALkDTG)
- Fed-RepVGG-B1-Inf (arch: RepVGG-B1-target-norepopt) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Eq1HajR9X29Dgi7mchrOxW0Bku1aNxlrK0Hh9WYdh6sj7g?e=KhmysH)
- Fed-CSLA-VGG-B1 (arch: RepVGG-B1-csla) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EmcEM8L5RGtNri_U98EAMBABKPMcSVUOxf3OnG8G7P_Rgw?e=VgrEag)
- FedRepOpt-VGG-B1 (arch: RepVGG-B1-target) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EnPBjdnsGoVBgSY6Pl4_3icBKbhzjv6gTZJ4rMBjf3fhyQ?e=JdNegq)

## Data Preparation
You can download our NIID Tiny ImageNet annotations files as follow:
- Cross silo NIID (&alpha;=0.1 in Dirichlet distribution, number of client=10) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Ep4Obqb1EYRItejHt4EZmQABCmYebaLf3-SxfUO51_SA0w?e=GWvPqJ)
- Cross device NIID (&alpha;=0.1 in Dirichlet distribution, number of client=100) [link](https://portland-my.sharepoint.com/:u:/g/personal/kinwailau6-c_my_cityu_edu_hk/EY2Gqx5p08hBqF9LR3ngxnYBsPfPuT9XUL3XM_nwhLWM8A?e=8hM2dU)

* `data_splitter/tiny-imagenet_json_splitter_direchlet.py` script provides a tool for generating IID and NIID annotations for Tiny-ImageNet.

## Preparation
* Requirements:
  * Python 3.8.0
  * PyTorch 1.7.1
  * Flower 1.3.0

* Install the required packages:
```
pip install -r requirements.txt
```

## Hyper-Search on CIFAR100
You can run the following command to conduct a hyper-parameter search on CIFAR100 in a centralized setting.
```
python -m torch.distributed.launch --nproc_per_node NUM_GPUS --master_port PORT_NUM main_repopt_centralized.py \
--data-path /path/to/cifar100 \
--arch ghost-hs \
--batch-size 128 \
--tag search \
--opts TRAIN.EPOCHS 600 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 10 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET cf100 TRAIN.CLIP_GRAD 5.0
```
* `hs.sh` provides examples of commands for finding hyperparameters for RepOpt-VGG-B1 and GhostNet.

## Train FedRepOpt on Tiny ImageNet
You can run the following command to conduct a federated training on Tiny ImageNet.
```
python src_fl/main.py \
--data-path /path/to/tiny-imagenet-200 \
--arch ghost-target-tinyImageNet \
--batch-size 32 \
--tag experiment \
--num_clients_per_round 10 \
--pool_size 10 \
--rounds 240 \
--scales-path /path/to/hyper-parameter-search/model \
--opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.01 TRAIN.LR_SCHEDULER.NAME step TRAIN.LR_SCHEDULER.DECAY_RATE 0.0 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 0 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET tiny_imagenet DATA.IMG_SIZE 64 LOGOUTPUT log TRAIN.OPTIMIZER.MOMENTUM 0.0 \
DATA.ANNOTATIONS_FED annotations_fed_alpha_0.1_clients_10 SEED 0
```
* `num_clients_per_round` represents number of clients participating in the training for each round and `pool_size` represents number of dataset partitions (= number of total clients). If `num_clients_per_round` is set to 10 and `pool_size` is 10, all the clients participate in the training.
* `round` represents the total number of FL rounds and `TRAIN.EPOCHS` represents the total number of training epochs for each clients.
* `train_repopt_fl.sh` provides training command examples for all the models.
* The evaluation results will be stored in `output/arch/server/log_rank0.txt`.


