import os
import json
import torch
import numpy as np
import random
import time
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings

class BDD100K_Night(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None,  multi_modal_vision=False,multi_modal_language=False, use_nlp=False, annotation_path=None):
        root = env_settings().bdd100k_dir if root is None else root
        super().__init__('bdd100k_night', root, image_loader)
        self.multi_modal_vision = multi_modal_vision

        self.img_pth = os.path.join(root, 'images/')
        self.anno_path = annotation_path or os.path.join(root, 'annotations/bdd100k_night.json')

        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        if not self.anno_path == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(self.anno_path, 'r') as f:
                dataset = json.load(f)
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
        self.sequence_list = self._get_sequence_list()
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))


    def _get_sequence_list(self):
        anns = {}
        for picture in self.dataset:
            timeofday = picture.get('attributes', {}).get('timeofday')
            if timeofday is not None and timeofday != 'night':
                continue
            for index, box in enumerate(picture.get('labels', [])):
                if 'box2d' not in box:
                    continue
                coords = box['box2d']
                if coords['x2'] <= coords['x1'] or coords['y2'] <= coords['y1']:
                    continue
                key = (picture['name'], index)
                anns[key] = dict(box, name=picture['name'])
        self.anns = anns

        seq_list = list(anns.keys())

        return seq_list

    def _get_anno(self, seq_id):
        anno = self.anns[self.sequence_list[seq_id]]
        return anno


    def _get_frames(self, seq_id):
        path = self.anns[self.sequence_list[seq_id]]['name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        if self.multi_modal_vision:
            img = np.concatenate((img, img), axis=-1)
        return img

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)

        x = anno['box2d']['x1']
        y = anno['box2d']['y1']
        width = anno['box2d']['x2'] - anno['box2d']['x1']
        height = anno['box2d']['y2'] - anno['box2d']['y1']

        bbox = torch.Tensor([x,y,width,height]).view(1, 4)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def is_video_sequence(self):
        return False

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # BDD100K is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta

    def get_name(self):
        return 'bdd100k_night'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_meta_info(self, seq_id):
        try:
            cat_dict_current = self.anns[self.sequence_list[seq_id]]['category']
            object_meta = OrderedDict({'object_class_name': cat_dict_current,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames
