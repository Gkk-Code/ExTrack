import json
import re
from pathlib import Path

import numpy as np
from .data import BaseDataset, Sequence, SequenceList


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(path))]


class RGBDataset(BaseDataset):
    """RGB sequences from sequence directories or an explicit sequences.json manifest."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.root = Path(getattr(self.env_settings, name + '_path'))
        if not self.root.is_dir():
            raise FileNotFoundError('Set %s_path in lib/test/evaluation/local.py: %s' % (name, self.root))

    def get_sequence_list(self):
        manifest = self.root / 'sequences.json'
        if manifest.is_file():
            with manifest.open(encoding='utf-8') as stream:
                entries = json.load(stream)
        else:
            entries = self._discover()
        if not entries:
            raise ValueError('No sequences found in %s; provide sequences.json for other layouts' % self.root)
        names = [entry['name'] for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError('Sequence names must be unique')
        return SequenceList([self._sequence(entry) for entry in entries])

    def _discover(self):
        entries = []
        for annotation in sorted(self.root.rglob('groundtruth*.txt'), key=natural_key):
            directory = annotation.parent
            if annotation.name not in ('groundtruth.txt', 'groundtruth_rect.txt'):
                continue
            image_dir = next((directory / folder for folder in ('img', 'imgs', 'color')
                              if (directory / folder).is_dir()), directory)
            entries.append(dict(name=directory.relative_to(self.root).as_posix(),
                                frames=image_dir.relative_to(self.root).as_posix(),
                                annotation=annotation.relative_to(self.root).as_posix()))
        if not entries and (self.root / 'anno').is_dir() and (self.root / 'data_seq').is_dir():
            entries = [dict(name=path.stem, frames='data_seq/' + path.stem,
                            annotation='anno/' + path.name)
                       for path in sorted((self.root / 'anno').glob('*.txt'), key=natural_key)]
        return entries

    def _sequence(self, entry):
        if isinstance(entry['frames'], list):
            frames = [str(self.root / frame) for frame in entry['frames']]
        else:
            frames = [str(path) for path in sorted((self.root / entry['frames']).iterdir(), key=natural_key)
                      if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
        annotation = self.root / entry['annotation']
        rows = [line.replace(',', ' ').split() for line in annotation.read_text(encoding='utf-8-sig').splitlines()
                if line.strip()]
        boxes = np.asarray(rows, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(frames) != len(boxes):
            raise ValueError('%s: require one x,y,w,h row per frame (%d frames, shape %s)' %
                             (entry['name'], len(frames), boxes.shape))
        if not frames or not all(Path(frame).is_file() for frame in frames):
            raise ValueError('Missing frames in %s' % entry['name'])
        return Sequence(entry['name'], frames, self.name, boxes)
