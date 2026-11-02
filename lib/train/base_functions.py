import torch
from torch.utils.data.distributed import DistributedSampler

from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, TrackingNet
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, TrackingNet_lmdb
from lib.train.dataset import BDD100K_Night, SHIFT_Night
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {key: cfg.DATA[key.upper()].FACTOR for key in ('template', 'search')}
    settings.output_sz = {key: cfg.DATA[key.upper()].SIZE for key in ('template', 'search')}
    settings.center_jitter_factor = {key: cfg.DATA[key.upper()].CENTER_JITTER for key in ('template', 'search')}
    settings.scale_jitter_factor = {key: cfg.DATA[key.upper()].SCALE_JITTER for key in ('template', 'search')}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.multi_modal_vision = cfg.DATA.MULTI_MODAL_VISION
    settings.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
    settings.use_nlp = cfg.DATA.USE_NLP
    settings.night_data = cfg.DATA.NIGHT
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if cfg.TRAIN.GLOBAL_BATCH_SIZE:
        if cfg.TRAIN.BATCH_SIZE < world_size or cfg.TRAIN.BATCH_SIZE % world_size:
            raise ValueError('Global BATCH_SIZE must be positive and divisible by the GPU count; '
                             'use GLOBAL_BATCH_SIZE=False for per-GPU batches.')
        settings.batchsize //= world_size
    if settings.batchsize < 1:
        raise ValueError('BATCH_SIZE must be positive')
    settings.fix_norm = False
    if settings.local_rank in (-1, 0):
        print('Batch size: %d per GPU, %d total (%d processes)' %
              (settings.batchsize, settings.batchsize * world_size, world_size))


def names2datasets(name_list, settings, image_loader):
    datasets = []
    for name in name_list:
        common = dict(image_loader=image_loader, multi_modal_vision=settings.multi_modal_vision)
        if name == 'BDD100K_NIGHT':
            root = settings.night_data.BDD_ROOT or settings.env.bdd100k_dir
            if not root:
                raise ValueError('Set DATA.NIGHT.BDD_ROOT or local.py bdd100k_dir')
            datasets.append(BDD100K_Night(root, annotation_path=settings.night_data.BDD_ANNOTATION or None, **common))
        elif name == 'SHIFT_NIGHT':
            root = settings.night_data.SHIFT_ROOT or settings.env.shift_dir
            if not root:
                raise ValueError('Set DATA.NIGHT.SHIFT_ROOT or local.py shift_dir')
            datasets.append(SHIFT_Night(root, annotation_path=settings.night_data.SHIFT_ANNOTATION or None, **common))
        else:
            common['multi_modal_language'] = settings.multi_modal_language
            if name == 'LASOT':
                cls = Lasot_lmdb if settings.use_lmdb else Lasot
                root = settings.env.lasot_lmdb_dir if settings.use_lmdb else settings.env.lasot_dir
                datasets.append(cls(root, split='train', use_nlp=settings.use_nlp.LASOT, **common))
            elif name in ('GOT10K_vottrain', 'GOT10K_votval', 'GOT10K_train_full'):
                cls = Got10k_lmdb if settings.use_lmdb else Got10k
                root = settings.env.got10k_lmdb_dir if settings.use_lmdb else settings.env.got10k_dir
                datasets.append(cls(root, split=name[len('GOT10K_'):], use_nlp=settings.use_nlp.GOT10K, **common))
            elif name == 'COCO17':
                cls = MSCOCOSeq_lmdb if settings.use_lmdb else MSCOCOSeq
                root = settings.env.coco_lmdb_dir if settings.use_lmdb else settings.env.coco_dir
                datasets.append(cls(root, version='2017', use_nlp=settings.use_nlp.COCO, **common))
            elif name == 'TRACKINGNET':
                cls = TrackingNet_lmdb if settings.use_lmdb else TrackingNet
                root = settings.env.trackingnet_lmdb_dir if settings.use_lmdb else settings.env.trackingnet_dir
                datasets.append(cls(root, use_nlp=settings.use_nlp.TRACKINGNET, **common))
            else:
                raise ValueError('Unsupported ExTrack training dataset: %s' % name)
    return datasets


def build_dataloaders(cfg, settings):
    settings.num_template = cfg.DATA.TEMPLATE.NUMBER
    settings.num_search = cfg.DATA.SEARCH.NUMBER
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(probability=0.5))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    data_processing = processing.ExTrackProcessing(
        search_area_factor=settings.search_area_factor, output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor, scale_jitter_factor=settings.scale_jitter_factor,
        mode='sequence', transform=transform_train, joint_transform=transform_joint,
        multi_modal_language=settings.multi_modal_language, settings=settings)
    active_task = getattr(settings, 'continual_task', 'day')
    if cfg.DATA.CONTINUAL.ENABLED:
        if active_task not in ('day', 'night'):
            raise ValueError('Unknown task: %s' % active_task)
        dataset_names = cfg.DATA.CONTINUAL[active_task.upper() + '_DATASETS_NAME']
        dataset_ratio = cfg.DATA.CONTINUAL[active_task.upper() + '_DATASETS_RATIO']
    else:
        dataset_names = cfg.DATA.TRAIN.DATASETS_NAME
        dataset_ratio = cfg.DATA.TRAIN.DATASETS_RATIO
    if not dataset_names or len(dataset_names) != len(dataset_ratio):
        raise ValueError('Provide one sampling weight for each training dataset')
    if any(weight < 0 for weight in dataset_ratio) or sum(dataset_ratio) <= 0:
        raise ValueError('Dataset weights must be nonnegative with a positive sum')
    if settings.local_rank in (-1, 0):
        print('Training task:', active_task, 'datasets:', dataset_names)
    dataset_train = sampler.TrackingSampler(
        datasets=names2datasets(dataset_names, settings, opencv_loader), p_datasets=dataset_ratio,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH, max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search, num_template_frames=settings.num_template,
        processing=data_processing, frame_sample_mode=cfg.DATA.SAMPLER_MODE,
        multi_modal_language=settings.multi_modal_language,
        history_length=cfg.MODEL.PROMPT.HISTORY_LENGTH if cfg.MODEL.PROMPT.ENABLED else None)
    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    return LTRLoader('train', dataset_train, training=True, batch_size=settings.batchsize,
                     shuffle=train_sampler is None, num_workers=cfg.TRAIN.NUM_WORKER,
                     drop_last=True, stack_dim=1, sampler=train_sampler)


def get_optimizer_scheduler(net, cfg):
    if cfg.TRAIN.TYPE != 'normal' or cfg.TRAIN.OPTIMIZER != 'ADAMW':
        raise ValueError('ExTrack uses full-network AdamW training (TRAIN.TYPE=normal)')
    param_dicts = [
        {'params': [p for name, p in net.named_parameters() if 'encoder' not in name and p.requires_grad]},
        {'params': [p for name, p in net.named_parameters() if 'encoder' in name and p.requires_grad],
         'lr': cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER}]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.DATA.CONTINUAL.ENABLED:
        period = cfg.DATA.CONTINUAL.EPOCHS_PER_TASK
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: cfg.TRAIN.SCHEDULER.DECAY_RATE **
            ((epoch % period) // cfg.TRAIN.LR_DROP_EPOCH))
    elif cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cfg.TRAIN.LR_DROP_EPOCH, gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    else:
        raise ValueError('Unsupported scheduler: %s' % cfg.TRAIN.SCHEDULER.TYPE)
    return optimizer, lr_scheduler
