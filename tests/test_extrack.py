"""Focused tests; run `python -m unittest discover -s tests -p test_extrack.py -v`."""

import importlib.util
import io
from pathlib import Path
import random
import tempfile
import unittest
import sys
import json
import copy
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PromptMemory = load_source('extrack_memory', 'lib/models/extrack/prompt_memory.py').PromptMemory
sample_ids = load_source('extrack_temporal', 'lib/train/data/temporal.py').sample_temporal_ids


def distributed_memory_worker(rank, filename):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(filename).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        memory = PromptMemory(8, 8, 2, 3, 2, 2)
        prompt = memory(torch.ones(2, 8) * (rank + 1))
        (prompt.square().mean() * (rank + 1)).backward()
        memory.update_memory(batch_size=2)
        for tensor in (memory.memory, memory.importance, memory.score_window):
            expected = tensor.clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(tensor, expected)
        assert int(memory.memory_count) == 3
    finally:
        dist.destroy_process_group()


def distributed_activation_worker(rank, filename):
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(filename).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        memory = PromptMemory(8, 8, 2, 3, 2, 2)
        model = DistributedDataParallel(memory, broadcast_buffers=False, find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for step in range(5):
            optimizer.zero_grad()
            memory.memory_enabled = step >= 2
            prompt = model(torch.randn(2, 8) + rank)
            prompt.square().mean().backward()
            optimizer.step()
            memory.update_memory(enabled=step >= 2, batch_size=2)
            assert int(memory.memory_count) == (3 if step >= 2 else 0)
        for tensor in (memory.memory, memory.importance, memory.score_window):
            expected = tensor.clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(tensor, expected)
    finally:
        dist.destroy_process_group()


def tiny_model():
    from lib.models.extrack.fastitpn import Fast_iTPN
    from lib.models.extrack.encoder import EncoderBase
    from lib.models.extrack.decoder import CenterPredictor
    from lib.models.extrack.extrack import EXTRACK
    body = Fast_iTPN(search_size=32, template_size=16, patch_size=16,
                    in_chans=6, embed_dim=32, depth_stage1=1, depth_stage2=1, depth=2,
                    num_heads=4, convmlp=True, cls_token=True, token_type_indicate=True)
    encoder = EncoderBase(body, True, [], 32)
    task_head = nn.Linear(32, 5).requires_grad_(False)
    model = EXTRACK(None, encoder, CenterPredictor(32, 32, feat_sz=2, stride=16),
                    task_head, num_template=2)
    model.prompt_memory = PromptMemory(32, 32, 5, 3, 4, 2)
    return model


class TemporalTests(unittest.TestCase):
    def test_sparse_ids_causal_and_max_gap(self):
        random.seed(3)
        numbers = [0, 100, 200, 500, 600]
        for _ in range(100):
            templates, current, history, mask = sample_ids([True] * 5, numbers, 6, 2, 200)
            t = current[0]
            for i in templates + [i for i, valid in zip(history, mask) if valid]:
                self.assertLessEqual(i, t)
                self.assertLessEqual(numbers[t] - numbers[i], 200)
            self.assertEqual([i for i, valid in zip(history, mask) if valid],
                             sorted(i for i, valid in zip(history, mask) if valid))

    def test_single_frame_is_appearance_only(self):
        templates, current, history, mask = sample_ids([True], None, 6, 2, 200)
        self.assertEqual(templates, [0, 0])
        self.assertEqual(current, [0])
        self.assertEqual(history, [0] * 6)
        self.assertFalse(any(mask))

    def test_invalid_order_rejected(self):
        with self.assertRaises(ValueError):
            sample_ids([True, True], [10, 5], 2, 1, 200)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.memory = PromptMemory(8, 8, 2, 3, 2, 2)

    def test_mean_residual_and_missing_history(self):
        observed = []
        hook = self.memory.motion_proj.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
        current = torch.ones(2, 8) * 7
        history = torch.stack((torch.ones(2, 8), torch.ones(2, 8) * 3), 1)
        out = self.memory(current, history)
        self.assertEqual(tuple(out.shape), (2, 2, 8))
        torch.testing.assert_close(observed[-1], torch.ones(2, 8) * 5)
        self.memory(current, history, torch.zeros(2, 2, dtype=torch.bool))
        torch.testing.assert_close(observed[-1], torch.zeros_like(current))
        hook.remove()

    def test_gradient_score_amp_and_top_m(self):
        for scale in (1., 1024.):
            memory = PromptMemory(8, 8, 2, 3, 2, 1)
            prompt = memory(torch.ones(2, 8))
            (prompt.sum() * scale / 2).backward()
            memory.update_memory(grad_scale=scale, batch_size=2)
            torch.testing.assert_close(memory.importance[:2], torch.full((2,), 8 ** .5))
            self.assertFalse(memory.memory.requires_grad)

    def test_fixed_scores_warmup_and_frozen_inference(self):
        prompt = self.memory(torch.ones(2, 8))
        prompt.square().mean().backward()
        self.memory.update_memory(enabled=False)
        self.assertEqual(int(self.memory.memory_count), 0)
        prompt = self.memory(torch.ones(2, 8))
        prompt.square().mean().backward()
        self.memory.update_memory(batch_size=2)
        old = {k: v.clone() for k, v in self.memory.state_dict().items()}
        self.memory.eval()
        with torch.no_grad():
            self.memory(torch.ones(2, 8))
            self.memory.update_memory()
        for key, value in self.memory.state_dict().items():
            torch.testing.assert_close(value, old[key])
        self.memory.reset_smoothing()
        torch.testing.assert_close(self.memory.importance, old['importance'])

    def test_checkpoint_roundtrip(self):
        prompt = self.memory(torch.ones(2, 8))
        prompt.square().sum().backward()
        self.memory.update_memory(batch_size=2)
        buffer = io.BytesIO()
        torch.save(self.memory.state_dict(), buffer)
        buffer.seek(0)
        restored = PromptMemory(8, 8, 2, 3, 2, 2)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        for key, value in self.memory.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])

    def test_two_process_pool_is_identical(self):
        if not dist.is_gloo_available():
            self.skipTest('gloo is unavailable')
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_memory_worker, args=(str(Path(directory) / 'init'),), nprocs=2)

    def test_ddp_warmup_write_and_retrieval(self):
        if not dist.is_gloo_available():
            self.skipTest('gloo is unavailable')
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_activation_worker, args=(str(Path(directory) / 'init'),), nprocs=2)


class BackboneTests(unittest.TestCase):
    def test_real_backbone_input_prompts_and_head_layout(self):
        from lib.models.extrack.fastitpn import Fast_iTPN
        backbone = Fast_iTPN(search_size=32, template_size=16, patch_size=16,
            in_chans=6, embed_dim=32, depth_stage1=1, depth_stage2=1, depth=2,
            num_heads=4, convmlp=True, cls_token=True, token_type_indicate=True)
        memory = PromptMemory(32, 32, 5, 3, 4, 1)
        lengths = []
        hook = backbone.blocks[-1].register_forward_pre_hook(lambda module, args: lengths.append(args[0].size(1)))
        search = [torch.randn(2, 6, 32, 32)]
        templates = [torch.randn(2, 6, 16, 16)] * 2
        boxes = [torch.tensor([[.1, .1, .5, .5]] * 2)] * 2
        output, descriptor = backbone(templates, search, boxes, None, None,
                                      prompt_generator=memory, return_descriptor=True)
        self.assertEqual(tuple(output[0].shape), (2, 7, 32))
        self.assertEqual(lengths, [12])
        self.assertEqual(tuple(descriptor.shape), (2, 32))
        output[0][:, 1:5].square().sum().backward()
        self.assertIsNotNone(memory._last_prompt.grad)
        memory.update_memory(batch_size=2)
        self.assertEqual(int(memory.memory_count), 2)
        hook.remove()

    def test_actor_loss_and_history_encoding(self):
        from lib.config.extrack.config import cfg as defaults
        from lib.train.actors.extrack import ExTrackActor
        from lib.utils.box_ops import giou_loss
        from lib.utils.focal_loss import FocalLoss
        cfg = copy.deepcopy(defaults)
        cfg.DATA.SEARCH.SIZE = 32
        model = tiny_model()
        actor = ExTrackActor(model, {'giou': giou_loss, 'l1': nn.functional.l1_loss,
                                    'focal': FocalLoss()},
                             {'giou': 2., 'l1': 5., 'focal': 1., 'task_cls': 0.},
                             SimpleNamespace(batchsize=2), cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        box = torch.tensor([[[.2, .2, .5, .5]] * 2])
        data = dict(template_images=torch.randn(2, 2, 6, 16, 16),
                    template_anno=box.repeat(2, 1, 1),
                    search_images=torch.randn(1, 2, 6, 32, 32), search_anno=box,
                    history_images=torch.randn(6, 2, 6, 32, 32),
                    history_valid=torch.zeros(6, 2, dtype=torch.bool),
                    dataset=['bdd100k_night', 'shift_night'])
        encoded = []
        original = model.encoder.body.image_descriptor
        def observe(images):
            encoded.append(images.shape[0])
            return original(images)
        with patch.object(model.encoder.body, 'image_descriptor', side_effect=observe):
            for epoch in (75, 76, 77, 1, 76):
                data['task_epoch'] = epoch
                data['history_valid'][-2:, 1] = epoch > 75
                optimizer.zero_grad()
                loss, status = actor(data)
                self.assertTrue(torch.isfinite(loss))
                self.assertIn('IoU', status)
                loss.backward()
                optimizer.step()
                actor.after_optimizer_step()
                if epoch == 75:
                    self.assertEqual(int(model.prompt_memory.memory_count), 0)
                    self.assertEqual(encoded, [])
                elif epoch > 75:
                    self.assertGreater(int(model.prompt_memory.memory_count), 0)
                    self.assertEqual(encoded[-1], 2)
            before = {k: v.clone() for k, v in model.prompt_memory.state_dict().items()}
            model.eval()
            with torch.no_grad():
                actor(data)
                actor.after_optimizer_step()
            for key, value in before.items():
                torch.testing.assert_close(model.prompt_memory.state_dict()[key], value)


class IntegrationTests(unittest.TestCase):
    def test_night_dataloader_with_training_transforms(self):
        import cv2
        import numpy as np
        from lib.config.extrack.config import cfg as defaults
        from lib.train.base_functions import build_dataloaders, update_settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'video').mkdir()
            image = np.full((96, 96, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / 'images' / 'night.jpg'), image))
            for name in ('000000.jpg', '000100.jpg', '000200.jpg'):
                self.assertTrue(cv2.imwrite(str(root / 'video' / name), image))
            box = dict(x1=10, y1=10, x2=70, y2=70)
            (root / 'bdd.json').write_text(json.dumps([
                dict(name='night.jpg', labels=[dict(id=1, category='car', box2d=box)])]))
            (root / 'shift.json').write_text(json.dumps({'video/7': dict(
                frame=['000000.jpg', '000100.jpg', '000200.jpg'],
                box2d=[box] * 3, category='car')}))
            cfg = copy.deepcopy(defaults)
            cfg.DATA.NIGHT.BDD_ROOT = cfg.DATA.NIGHT.SHIFT_ROOT = directory
            cfg.DATA.NIGHT.BDD_ANNOTATION = str(root / 'bdd.json')
            cfg.DATA.NIGHT.SHIFT_ANNOTATION = str(root / 'shift.json')
            cfg.TRAIN.BATCH_SIZE = 2
            cfg.TRAIN.NUM_WORKER = 0
            cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 4
            cfg.DATA.SEARCH.SIZE = 32
            cfg.DATA.TEMPLATE.SIZE = 16
            cfg.DATA.SEARCH.CENTER_JITTER = cfg.DATA.SEARCH.SCALE_JITTER = 0.
            settings = SimpleNamespace(local_rank=-1, continual_task='night',
                                       env=SimpleNamespace(), use_lmdb=False)
            update_settings(settings, cfg)
            for dataset_name, expected_history in [('BDD100K_NIGHT', False), ('SHIFT_NIGHT', True)]:
                cfg.DATA.CONTINUAL.NIGHT_DATASETS_NAME = [dataset_name]
                cfg.DATA.CONTINUAL.NIGHT_DATASETS_RATIO = [1]
                loader = build_dataloaders(cfg, settings)
                self.assertEqual(len(loader), 2)
                batch = next(iter(loader))
                self.assertEqual(tuple(batch['template_images'].shape), (2, 2, 6, 16, 16))
                self.assertEqual(tuple(batch['search_images'].shape), (1, 2, 6, 32, 32))
                self.assertEqual(tuple(batch['history_images'].shape), (6, 2, 6, 32, 32))
                self.assertEqual(tuple(batch['history_valid'].shape), (6, 2))
                self.assertEqual(bool(batch['history_valid'].any()), expected_history)

    def test_task_transition_preserves_model_optimizer_and_memory(self):
        from lib.train.trainers.ltr_trainer import LTRTrainer
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        prompt = model.prompt_memory(torch.randn(2, 32))
        prompt.square().mean().backward()
        optimizer.step()
        model.prompt_memory.update_memory(batch_size=2)
        before = {key: value.clone() for key, value in model.state_dict().items()
                  if not key.startswith('prompt_memory.score_')}
        opt_before = copy.deepcopy(optimizer.state_dict())
        trainer = object.__new__(LTRTrainer)
        trainer.actor = SimpleNamespace(net=model)
        trainer.optimizer = optimizer
        trainer.settings = SimpleNamespace(epochs_per_task=150, continual_task='day', local_rank=1)
        old_loader = SimpleNamespace(name='day', epoch_interval=1, sampler=None)
        new_loader = SimpleNamespace(name='night', epoch_interval=1, sampler=None)
        trainer.loaders = [old_loader]
        trainer.loader_factory = lambda: new_loader
        observed = []
        trainer.cycle_dataset = lambda loader: observed.append(loader.name)
        trainer._stats_new_epoch = lambda: None
        trainer.epoch = 151
        trainer.train_epoch()
        self.assertEqual(observed, ['night'])
        self.assertEqual(trainer.settings.continual_task, 'night')
        self.assertIs(trainer.optimizer, optimizer)
        self.assertEqual(int(model.prompt_memory.score_steps), 0)
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        for index, state in opt_before['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(optimizer.state_dict()['state'][index][key], value)

    def test_batch_size_for_one_two_four_five_processes(self):
        from lib.config.extrack.config import cfg as defaults
        from lib.train.base_functions import update_settings
        for world_size in (1, 2, 4, 5):
            cfg = copy.deepcopy(defaults)
            settings = SimpleNamespace(local_rank=1)
            with patch('torch.distributed.is_initialized', return_value=True), \
                    patch('torch.distributed.get_world_size', return_value=world_size):
                update_settings(settings, cfg)
                self.assertEqual(settings.batchsize, 16)
                cfg.TRAIN.GLOBAL_BATCH_SIZE = True
                if world_size == 5:
                    with self.assertRaises(ValueError):
                        update_settings(settings, cfg)
                    cfg.TRAIN.BATCH_SIZE = 20
                    update_settings(settings, cfg)
                    self.assertEqual(settings.batchsize, 4)
                else:
                    update_settings(settings, cfg)
                    self.assertEqual(settings.batchsize * world_size, 16)

    def test_dataset_and_static_processing(self):
        import numpy as np
        from lib.train.dataset import BDD100K_Night, SHIFT_Night
        from lib.train.data.sampler import TrackingSampler
        from lib.train.data.processing import ExTrackProcessing
        import lib.train.data.transforms as transforms
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / 'bdd.json'
            box = dict(x1=10, y1=10, x2=70, y2=70)
            records = [dict(name='night.jpg', attributes={'timeofday': 'night'},
                            labels=[dict(id=1, category='car', box2d=box)]),
                       dict(name='day.jpg', attributes={'timeofday': 'daytime'},
                            labels=[dict(id=1, category='car', box2d=box)])]
            annotation.write_text(json.dumps(records), encoding='utf-8')
            image_loader = lambda path: np.full((96, 96, 3), 128, dtype=np.uint8)
            dataset = BDD100K_Night(directory, image_loader=image_loader,
                                   multi_modal_vision=True, annotation_path=str(annotation))
            self.assertEqual(len(dataset.sequence_list), 1)
            self.assertFalse(dataset.is_video_sequence())
            processing = ExTrackProcessing(
                {'search': 2., 'template': 2.}, {'search': 32, 'template': 16},
                {'search': 0., 'template': 0.}, {'search': 0., 'template': 0.},
                mode='sequence', transform=transforms.Transform(transforms.ToTensorAndJitter(0.2)))
            sampler = TrackingSampler([dataset], [1], 10, 200, 1, 2,
                                      processing=processing, history_length=6)
            sample = sampler[0]
            self.assertEqual(tuple(sample['history_images'].shape), (6, 6, 32, 32))
            self.assertFalse(sample['history_valid'].any())
            torch.testing.assert_close(sample['history_images'], sample['search_images'].expand(6, -1, -1, -1))
            shift_file = root / 'shift.json'
            shift_file.write_text(json.dumps({'video/7': dict(frame=['000200.jpg', '000000.jpg'],
                                      box2d=[box, box], category='car')}), encoding='utf-8')
            shift = SHIFT_Night(directory, image_loader=image_loader, annotation_path=str(shift_file))
            self.assertEqual(shift.get_sequence_info(0)['frame_numbers'].tolist(), [0, 200])
            sample = TrackingSampler([shift], [1], 10, 200, 1, 2, history_length=6)[0]
            self.assertEqual(sample['history_valid'].sum().item(), 1)

    def test_rgb_adapter_frame_order_and_manifest(self):
        from lib.test.evaluation.rgbdataset import RGBDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'bag' / 'img'
            images.mkdir(parents=True)
            for name in ('10.jpg', '2.jpg', '1.jpg'):
                (images / name).touch()
            (images.parent / 'groundtruth.txt').write_text('1,2,3,4\n' * 3)
            with patch('lib.test.evaluation.data.env_settings',
                       return_value=SimpleNamespace(llot_path=directory)):
                sequences = RGBDataset('llot').get_sequence_list()
                self.assertEqual([Path(frame).name for frame in sequences[0].frames], ['1.jpg', '2.jpg', '10.jpg'])
                (root / 'sequences.json').write_text(json.dumps([
                    dict(name='bag_1', frames='bag/img', annotation='bag/groundtruth.txt')]))
                self.assertEqual(RGBDataset('llot').get_sequence_list()[0].name, 'bag_1')
                (images.parent / 'groundtruth.txt').write_text('1,2,3,4\n')
                with self.assertRaises(ValueError):
                    RGBDataset('llot').get_sequence_list()

    def test_trainer_resume_preserves_optimizer_scheduler_and_memory(self):
        from lib.train.trainers.base_trainer import BaseTrainer
        from lib.train.actors.base_actor import BaseActor
        from lib.config.extrack.config import cfg as defaults
        from lib.train.base_functions import get_optimizer_scheduler
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(env=SimpleNamespace(workspace_dir=directory),
                                       save_dir=directory, local_rank=-1, device=torch.device('cpu'),
                                       use_gpu=False, project_path='train/extrack/test')
            model = tiny_model()
            optimizer, scheduler = get_optimizer_scheduler(model, copy.deepcopy(defaults))
            prompt = model.prompt_memory(torch.randn(2, 32))
            prompt.square().mean().backward()
            optimizer.step()
            scheduler.step()
            model.prompt_memory.update_memory(batch_size=2)
            trainer = BaseTrainer(BaseActor(model, None), [], optimizer, settings, scheduler)
            trainer.epoch = 150
            trainer.save_checkpoint()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            path = Path(directory) / 'checkpoints' / settings.project_path / 'EXTRACK_ep0150.pth.tar'
            state = torch.load(path, weights_only=False)
            state['net_type'] = 'SUTRACK'
            torch.save(state, path)
            restored = tiny_model()
            opt2, sched2 = get_optimizer_scheduler(restored, copy.deepcopy(defaults))
            resumed = BaseTrainer(BaseActor(restored, None), [], opt2, settings, sched2)
            self.assertTrue(resumed.load_checkpoint(str(path.parent)))
            self.assertEqual(resumed.epoch, 150)
            self.assertEqual(sched2.state_dict(), scheduler.state_dict())
            self.assertEqual(len(opt2.state), len(optimizer.state))
            for key, value in before.items():
                torch.testing.assert_close(restored.state_dict()[key], value)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
