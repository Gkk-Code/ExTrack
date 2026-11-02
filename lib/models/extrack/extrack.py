"""
ExTrack Model
"""
import torch
import math
from torch import nn
from .encoder import build_encoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from .prompt_memory import PromptMemory
from lib.utils.box_ops import box_xyxy_to_cxcywh


class EXTRACK(nn.Module):
    """ This is the base class for ExTrack """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average"):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        self.prompt_memory = None

    @torch.no_grad()
    def update_prompt_memory(self, enabled=True, **kwargs):
        if self.prompt_memory is not None:
            self.prompt_memory.update_memory(enabled=enabled, **kwargs)


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder", history_images=None, history_features=None,
                history_mask=None, memory_enabled=True, task_supervision=True):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index,
                                        history_images, history_features, history_mask)
        elif mode == "track":
            if text_data is not None:
                text_src = self.forward_textencoder(text_data)
            if self.prompt_memory is not None:
                self.prompt_memory.memory_enabled = memory_enabled
            features = self.forward_encoder(template_list, search_list, template_anno_list, text_src,
                                            task_index, history_images, history_features, history_mask)
            out = self.forward_decoder(features)
            if task_supervision:
                out['task_class'] = self.forward_task_decoder(features)
            return out
        elif mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        # Forward the encoder
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index,
                        history_images=None, history_features=None, history_mask=None,
                        return_descriptor=False):
        if self.prompt_memory is None:
            return self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        if history_images is not None:
            batch, steps = history_images.shape[:2]
            valid = (history_mask.bool() if history_mask is not None else
                     torch.ones(batch, steps, dtype=torch.bool, device=history_images.device))
            with torch.no_grad():
                history_features = history_images.new_zeros(batch, steps, self.encoder.num_channels)
                if valid.any():
                    descriptor = self.encoder.body.image_descriptor(history_images[valid])
                    history_features[valid] = descriptor.to(history_features.dtype)
        return self.encoder(template_list, search_list, template_anno_list, text_src, task_index,
                            prompt_generator=self.prompt_memory, history_features=history_features,
                            history_mask=history_mask, return_descriptor=return_descriptor)

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature

def build_extrack(cfg):
    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        from .clip import build_textencoder
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    if cfg.TRAIN.TASK_CE_WEIGHT == 0:
        task_decoder.requires_grad_(False)
    model = EXTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )
    prompt_cfg = getattr(cfg.MODEL, "PROMPT", None)
    if prompt_cfg is not None and getattr(prompt_cfg, "ENABLED", False):
        model.prompt_memory = PromptMemory(
            feature_dim=encoder.num_channels,
            prompt_dim=getattr(prompt_cfg, "DIM", 512),
            prompt_length=getattr(prompt_cfg, "LENGTH", 5),
            memory_size=getattr(prompt_cfg, "MEMORY_SIZE", 50),
            num_heads=getattr(prompt_cfg, "NUM_HEADS", 8),
            smoothing_window=getattr(prompt_cfg, "SMOOTHING_WINDOW", 5))
        if not hasattr(encoder.body, 'image_descriptor'):
            raise ValueError("ExTrack input prompts require the fastitpn backbone")

    return model
