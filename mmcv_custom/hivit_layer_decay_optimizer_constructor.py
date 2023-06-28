# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from BEiT library:
https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py
"""

import json

from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmdet.utils import get_root_logger
from mmcv.runner import get_dist_info


def get_hivit_layer_id(name, num_layers, main_block):
    if name.startswith('backbone.'):
        name = name[9:]
        if name in ['cls_token', 'pos_embed', 'absolute_pos_embed']:
            return 0
        elif name.startswith('patch_embed'):
            return 0
        elif name.startswith('blocks'):
            i = int(name.split('.')[1])
            while i >= 0:
                try:
                    return main_block.index(i) + 1
                except ValueError:
                    i = i - 1
            return 0
    return num_layers - 1


@OPTIMIZER_BUILDERS.register_module()
class HiViTLayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def _validate_cfg(self):
        pass

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = get_root_logger()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        if isinstance(self.paramwise_cfg, dict):
            num_layers = self.paramwise_cfg.get('num_layers') + 2
            decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        else:
            num_layers, decay_rate = self.paramwise_cfg
            num_layers = num_layers + 2
        logger.info('Build HiViTLayerDecayOptimizerConstructor')

        weight_decay = self.base_wd
        main_block = [
            i for i, blk in enumerate(module.backbone.blocks)
            if hasattr(blk, 'attn') and blk.attn is not None
        ]
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token', 'absolute_pos_embed') or 'relative_position_bias_table' in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_hivit_layer_id(name, num_layers, main_block)
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())
