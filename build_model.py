
from optimizer import build_optimizer

def build_model(config):
    if 'vgg' in config.MODEL.ARCH.lower() or 'B1' in config.MODEL.ARCH or 'B2' in config.MODEL.ARCH or 'L1' in config.MODEL.ARCH or 'L2' in config.MODEL.ARCH:
        from repoptimizer.repoptvgg_model import RepOptVGG
       
        if 'B0' in config.MODEL.ARCH:
            num_blocks = [2, 3, 8, 1]
            width_multiplier = [2, 2, 2, 4]
        elif 'B1' in config.MODEL.ARCH:
            num_blocks = [4, 6, 16, 1]
            width_multiplier = [2, 2, 2, 4]
        elif 'B2' in config.MODEL.ARCH:
            num_blocks = [4, 6, 16, 1]
            width_multiplier = [2.5, 2.5, 2.5, 5]
        elif 'L1' in config.MODEL.ARCH:
            num_blocks = [8, 14, 24, 1]
            width_multiplier = [2, 2, 2, 4]
        elif 'L2' in config.MODEL.ARCH:
            num_blocks = [8, 14, 24, 1]
            width_multiplier = [2.5, 2.5, 2.5, 5]
        else:
            raise ValueError('Not yet supported. You may add the architectural settings here.')
        
        if '-repvgg' in config.MODEL.ARCH:
            #   as baseline
            if config.DATA.DATASET == 'imagenet':
                model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='repvgg', num_classes=1000)
            elif config.DATA.DATASET == 'tiny_imagenet':
                model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='repvgg', num_classes=200)
            else:
                raise ValueError('not supported')
            optimizer = build_optimizer(config, model)

        elif '-csla' in config.MODEL.ARCH:
            if config.DATA.DATASET == 'imagenet':
                model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='csla', num_classes=1000)
            elif config.DATA.DATASET == 'tiny_imagenet':
                model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='csla', num_classes=200)
            else:
                raise ValueError('not supported')
            optimizer = build_optimizer(config, model)
        
        elif '-target' in config.MODEL.ARCH:
            #   build target model
            if config.EVAL_MODE or '-norepopt' in config.MODEL.ARCH:
                if config.DATA.DATASET == 'imagenet':
                    model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='target', num_classes=1000)
                elif config.DATA.DATASET == 'tiny_imagenet':
                    model = RepOptVGG(num_blocks=num_blocks, width_multiplier=width_multiplier, mode='target', num_classes=200)
                else:
                    raise ValueError('not supported dataset')
                optimizer = build_optimizer(config, model)  # just a placeholder for testing or the ablation study with regular optimizer for training
            else:
                from repoptimizer.repoptvgg_impl import build_RepOptVGG_and_SGD_optimizer_from_pth
                if config.DATA.DATASET == 'imagenet':
                    model, optimizer = build_RepOptVGG_and_SGD_optimizer_from_pth(num_blocks, width_multiplier, config.TRAIN.SCALES_PATH,
                                                lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY,
                                                num_classes=1000)
                elif config.DATA.DATASET == 'tiny_imagenet':
                    model, optimizer = build_RepOptVGG_and_SGD_optimizer_from_pth(num_blocks, width_multiplier, config.TRAIN.SCALES_PATH,
                                                lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY,
                                                num_classes=200)
                else:
                    raise ValueError('not supported dataset')
        else:
            raise ValueError('not supported')

    elif 'ghost' in config.MODEL.ARCH.lower():

        from repoptimizer.repoptghostnet_model import repoptghostnet_0_5x

        repoptghostnet = repoptghostnet_0_5x
        
        if '-rep' in config.MODEL.ARCH:
            #   as baseline
            if config.DATA.DATASET == 'imagenet':
                model = repoptghostnet(mode='rep', num_classes=1000)
            elif config.DATA.DATASET == 'tiny_imagenet':
                model = repoptghostnet(mode='rep', num_classes=200)
            else:
                raise ValueError('not supported')
            optimizer = build_optimizer(config, model)

        elif '-csla' in config.MODEL.ARCH:
            if config.DATA.DATASET == 'imagenet':
                model = repoptghostnet(mode='csla', num_classes=1000)
            elif config.DATA.DATASET == 'tiny_imagenet':
                model = repoptghostnet(mode='csla', num_classes=200)
            else:
                raise ValueError('not supported')
            optimizer = build_optimizer(config, model)

        elif '-original' in config.MODEL.ARCH:
            if config.DATA.DATASET == 'imagenet':
                model = repoptghostnet(mode='org', num_classes=1000)
            elif config.DATA.DATASET == 'tiny_imagenet':
                model = repoptghostnet(mode='org', num_classes=200)
            else:
                raise ValueError('not supported')
            optimizer = build_optimizer(config, model)

        elif '-target' in config.MODEL.ARCH:
            #   build target model
            if config.EVAL_MODE or '-norepopt' in config.MODEL.ARCH:
                if config.DATA.DATASET == 'imagenet':
                    model = repoptghostnet(mode='target', num_classes=1000)
                elif config.DATA.DATASET == 'tiny_imagenet':
                    model = repoptghostnet(mode='target', num_classes=200)
                else:
                    raise ValueError('not supported')
                optimizer = build_optimizer(config, model)  # just a placeholder for testing or the ablation study with regular optimizer for training
            else:
                from repoptimizer.repoptghostnet_impl import build_RepOptGhostNet_and_SGD_optimizer_from_pth
                if config.DATA.DATASET == 'imagenet':
                    model, optimizer = build_RepOptGhostNet_and_SGD_optimizer_from_pth(repoptghostnet, config.TRAIN.SCALES_PATH,
                                                lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY,
                                                num_classes=1000)
                elif config.DATA.DATASET == 'tiny_imagenet':
                    model, optimizer = build_RepOptGhostNet_and_SGD_optimizer_from_pth(repoptghostnet, config.TRAIN.SCALES_PATH,
                                                lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY,
                                                num_classes=200)
                else:
                    raise ValueError('not supported')
                                                
        else:
            raise ValueError('not supported')
    
    else:
        raise ValueError('TODO: support other models except for RepOpt-VGG.')

    return model, optimizer