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

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.models.base_model import BaseModel
import ipdb

class FasterRCNN(BaseModel):
    def __init__(self, device, base, n_classes = 1000, pretrained = True, retrain_base = False):
        """Initialize the model

        Args:
            device: the device (cpu/gpu) to place the tensors
            base: (str) name of base model
            n_classes: (int) number of classes
        """
        super().__init__(device)

        self.n_classes = n_classes
        self.base = base
        self.retrain_base = retrain_base
        self.pretrained = pretrained
        self.build_model()

    def set_parameter_requires_grad(self, is_features_extracter):
        """Set requires grad for all model's parameters
        Args:
            is_feature_extracter: (bool) if we only use base model as feature extractor (no training)
        """
        if is_features_extracter:
            for param in self.model.parameters():
                param.requires_grad = False

    def build_model(self):
        """Build model architecture
        """
        # load an instance segmentation model pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_classes)



    def forward(self, input_tensor, target = None):
        """Forward function of the model
        Args:
            input_tensor: pytorch input tensor
        """
        return self.model(input_tensor, target)
