import numpy as np
import torch
from torch import nn
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args

REGISTERED_IMG_RPN_CLASSES = {}

def register_img_rpn(cls, name=None):
    global REGISTERED_IMG_RPN_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_IMG_RPN_CLASSES, f"exist class: {REGISTERED_IMG_RPN_CLASSES}"
    REGISTERED_IMG_RPN_CLASSES[name] = cls
    return cls

def get_img_rpn_class(name):
    global REGISTERED_IMG_RPN_CLASSES
    assert name in REGISTERED_IMG_RPN_CLASSES, f"available class: {REGISTERED_IMG_RPN_CLASSES}"
    return REGISTERED_IMG_RPN_CLASSES[name]

@register_img_rpn
class img_extractor_VGG16(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 img_input_channel=3,
                 img_extractor_layer_nums=[2, 3],
                 layer_strides=[2, 2],
                 num_filters=[32, 64],
                 upsample_strides=[1, 2],
                 num_upsample_filters=[128, 128],
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 name='img_extractor_SSD_like'):
        super(img_extractor_VGG16, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        assert len(layer_strides) == len(img_extractor_layer_nums)
        assert len(num_filters) == len(img_extractor_layer_nums)
        assert len(upsample_strides) == len(img_extractor_layer_nums)
        assert len(num_upsample_filters) == len(img_extractor_layer_nums)
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)

        self.block1 = Sequential()
        for i in range(img_extractor_layer_nums[0]):
            if i == 0:
                block1_in = 3
            else:
                block1_in = num_filters[0]
            self.block1.add(
                Conv2d(block1_in, num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU(inplace=False))
        self.block1.add(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = Sequential()
        for i in range(img_extractor_layer_nums[1]):
            if i == 0:
                block2_in = num_filters[0]
            else:
                block2_in = num_filters[1]
            self.block2.add(
                Conv2d(block2_in, num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU(inplace=False))
        self.block2.add(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.block3 = Sequential()
        for i in range(img_extractor_layer_nums[2]):
            if i == 0:
                block2_in = num_filters[1]
            else:
                block2_in = num_filters[2]
            self.block2.add(
                Conv2d(block2_in, num_filters[2], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[2]))
            self.block2.add(nn.ReLU(inplace=False))
        self.block2.add(torch.nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, inputs, bev=None):  # x: [1, 3, 375, 1240]
        img_feat_block1 = self.block1(inputs)      # [1, 64, 188, 620]
        img_feat_block2 = self.block2(img_feat_block1)      # [1,128, 94, 310]
        img_feat_block3 = self.block3(img_feat_block2)
        return img_feat_block3