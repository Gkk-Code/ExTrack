import os
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from lib.train.trainers import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from .base_functions import *
from lib.models.extrack import build_extrack
from lib.train.actors import ExTrackActor
from lib.utils.focal_loss import FocalLoss
import importlib
import glob
from lib.utils.checkpoint import load_training_checkpoint


def run(settings):
    settings.description = 'ExTrack sequential Day-to-Night training'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    continual = cfg.DATA.CONTINUAL.ENABLED
    requested_task = getattr(settings, 'continual_task', None)
    if continual:
        settings.epochs_per_task = cfg.DATA.CONTINUAL.EPOCHS_PER_TASK
        settings.continual_task = requested_task or 'day'
        if cfg.TRAIN.EPOCH != 2 * settings.epochs_per_task:
            raise ValueError("Sequential Day->Night training requires EPOCH = 2 * EPOCHS_PER_TASK")
        if requested_task == 'night' and not cfg.TRAIN.RESUME:
            raise ValueError("Night-only continuation requires TRAIN.RESUME with the daytime checkpoint")
        checkpoint = cfg.TRAIN.RESUME
        if checkpoint and os.path.isdir(os.path.expanduser(checkpoint)):
            candidates = sorted(glob.glob(os.path.join(os.path.expanduser(checkpoint), '*_ep*.pth.tar')))
            if not candidates:
                raise FileNotFoundError('No training checkpoint in %s' % checkpoint)
            checkpoint = candidates[-1]
            cfg.TRAIN.RESUME = checkpoint
        if not checkpoint:
            save_root = settings.save_dir or settings.env.workspace_dir
            candidates = sorted(glob.glob(os.path.join(save_root, 'checkpoints',
                                settings.project_path, 'EXTRACK_ep*.pth.tar')))
            checkpoint = candidates[-1] if candidates else None
        if checkpoint:
            resume_epoch = load_training_checkpoint(checkpoint)['epoch']
            settings.continual_task = 'night' if resume_epoch >= settings.epochs_per_task else 'day'

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")


    # Create network
    if settings.script_name == "extrack":
        net = build_extrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if cfg.TRAIN.PRETRAINED_PATH:
        state = load_training_checkpoint(cfg.TRAIN.PRETRAINED_PATH)
        weights = state.get('net', state)
        # Initialization is distinct from resuming an ExTrack optimizer/memory.
        weights = {k: v for k, v in weights.items() if not k.startswith('prompt_memory.')}
        missing, unexpected = net.load_state_dict(weights, strict=False)
        invalid = [k for k in missing if not k.startswith('prompt_memory.')]
        if invalid or unexpected:
            raise ValueError("Incompatible initialization checkpoint: %s %s" % (invalid, unexpected))
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    if settings.local_rank != -1:
        # Retrieval parameters become active only after memory has been populated.
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # Loss functions and Actors
    if  settings.script_name == "extrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = ExTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    if continual:
        trainer.loader_factory = lambda: build_dataloaders(cfg, settings)

    # train
    if cfg.TRAIN.RESUME:
        trainer.load_checkpoint(cfg.TRAIN.RESUME)
        if requested_task == 'night' and trainer.epoch < settings.epochs_per_task:
            raise ValueError("Complete the daytime task before requesting night-only continuation")
    final_epoch = settings.epochs_per_task if continual and requested_task == 'day' else cfg.TRAIN.EPOCH
    trainer.train(final_epoch, load_latest=not bool(cfg.TRAIN.RESUME), fail_safe=False)
