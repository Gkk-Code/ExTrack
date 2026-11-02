import importlib
from pathlib import Path


class EnvSettings:
    def __init__(self, workspace_dir=None, data_dir=None, save_dir=None):
        project = Path(workspace_dir) if workspace_dir else Path(__file__).resolve().parents[3]
        data = Path(data_dir) if data_dir else project / 'data'
        output = Path(save_dir) if save_dir else project / 'output'
        self.prj_dir = str(project)
        self.save_dir = str(output)
        self.results_path = str(output / 'test' / 'tracking_results')
        self.segmentation_path = str(output / 'test' / 'segmentation_results')
        self.network_path = str(output / 'test' / 'networks')
        self.result_plot_path = str(output / 'test' / 'result_plots')
        for name, folder in dict(otb='OTB2015', lasot='lasot', got10k='got10k',
                                 lasot_lmdb='lasot_lmdb', lasot_extension_subset='lasot_extension_subset',
                                 trackingnet='trackingnet', dtb70='DTB70', visdrone2018='VisDrone2018',
                                 uavdt='UAVDT', uavdark135='UAVDark135', nat2024_1='NAT2024-1',
                                 darktrack2021='DarkTrack2021', llot='LLOT').items():
            setattr(self, name + '_path', str(data / folder))
        self.got_packed_results_path = str(output / 'got10k_packed')
        self.got_reports_path = str(output / 'got10k_reports')
        self.tn_packed_results_path = str(output / 'trackingnet_packed')


def create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir):
    with Path(__file__).with_name('local.py').open('w', encoding='utf-8') as stream:
        stream.write('from lib.test.evaluation.environment import EnvSettings\n\n\n'
                     'def local_env_settings():\n    settings = EnvSettings()\n')
        for key, value in vars(EnvSettings(workspace_dir, data_dir, save_dir)).items():
            stream.write('    settings.%s = %r\n' % (key, value))
        stream.write('    return settings\n')


def create_default_local_file():
    create_default_local_file_ITP_test(None, None, None)


def env_settings():
    return importlib.import_module('lib.test.evaluation.local').local_env_settings()
