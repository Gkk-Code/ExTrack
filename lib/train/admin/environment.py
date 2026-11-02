import importlib
from pathlib import Path


def default_paths(workspace_dir=None, data_dir=None):
    project = Path(__file__).resolve().parents[3]
    workspace = Path(workspace_dir) if workspace_dir else project / 'output'
    data = Path(data_dir) if data_dir else project / 'data'
    return dict(workspace_dir=str(workspace), tensorboard_dir=str(workspace / 'tensorboard'),
                pretrained_networks=str(project / 'pretrained'), lasot_dir=str(data / 'lasot'),
                got10k_dir=str(data / 'got10k' / 'train'), trackingnet_dir=str(data / 'trackingnet'),
                coco_dir=str(data / 'coco'), bdd100k_dir=str(data / 'bdd100k'), shift_dir=str(data / 'shift'),
                lasot_lmdb_dir=str(data / 'lasot_lmdb'), got10k_lmdb_dir=str(data / 'got10k_lmdb'),
                trackingnet_lmdb_dir=str(data / 'trackingnet_lmdb'), coco_lmdb_dir=str(data / 'coco_lmdb'))


def create_default_local_file_ITP_train(workspace_dir, data_dir):
    path = Path(__file__).with_name('local.py')
    with path.open('w', encoding='utf-8') as stream:
        stream.write('class EnvironmentSettings:\n    def __init__(self):\n')
        for key, value in default_paths(workspace_dir, data_dir).items():
            stream.write('        self.%s = %r\n' % (key, value))


def create_default_local_file():
    create_default_local_file_ITP_train(None, None)


def env_settings():
    return importlib.import_module('lib.train.admin.local').EnvironmentSettings()
