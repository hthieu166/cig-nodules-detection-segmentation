"""Dummy model"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from torch import nn
import torchvision
import torchvision.models as models 
import warnings
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN as torchFasterRCNN
from src.models.base_model import BaseModel
from src.models.detectors.fpn_backbone import get_fpn_backbone
import ipdb
class FasterRCNN(BaseModel):
    def __init__(self, device, backbone = None, n_classes = 1000, pretrained = True, \
                anchor_size = None , aspect_ratios = None):
        """Initialize the model

        Args:
            device: the device (cpu/gpu) to place the tensors
            backbone: (str) name of backbone model, by default, 'ResNet 50' is used 
            n_classes: (int) number of classes (rmb to +1 (for background cls))
            pretrained: (bool) pretrained model on ImageNet/COCO is used
            anchorsize: (tuple) all anchor sizes for RPN
            aspect_ratios: (tuple) all ratios for RPN
         """

        super().__init__(device)
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.support_backbones = ['resnet50', 'resnet101', 'mobilenet-v2']
        self.backbone = backbone
        self.anchor_size   = anchor_size
        self.aspect_ratios = aspect_ratios
        self.rpn_anchor_generator = None
        self.build_model()

    def build_model(self):
        """Build model architecture
        """
        # Setup RPN Anchors Generator
        if (self.anchor_size != None) and (self.aspect_ratios != None):
            anchor_sizes = tuple([tuple([anc]) for anc in self.anchor_size])
            aspect_ratios = (tuple(self.aspect_ratios),) * len(anchor_sizes)
            self.rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        
        # Build main backbone
        if (self.backbone == None) or (self.backbone == 'resnet50'):
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=self.pretrained,
                rpn_anchor_generator = self.rpn_anchor_generator
                )
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_classes)
        else:
            assert self.backbone in self.support_backbones, "Backbone %s does not support" % self.backbone
            if (self.pretrained):
                warnings.warn("Pretrained model on COCO is not available, pretrained on Image is used instead. \
                            Note that the performance can be be affected heavily")
            self.backbone = get_fpn_backbone(self.backbone, self.pretrained)
            self.model = torchFasterRCNN(self.backbone, num_classes=self.n_classes)
            
    def forward(self, input_tensor, target = None):
        """Forward function of the model
        Args:
            input_tensor: pytorch input tensor
        """ 
        ipdb.set_trace()
        return self.model(input_tensor, target)
