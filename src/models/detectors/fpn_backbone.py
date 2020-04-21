from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from src.models.lib.efficientnet.efficientnet import EfficientNet
import ipdb
import torchvision.models as models
from torchvision.ops import misc as misc_nn_ops

def resnet_fpn_backbone(backbone_name, pretrained):
    if (backbone_name == 'resnet50'):
        backbone = models.resnet50(pretrained = pretrained)
    elif (backbone_name == 'resnet101'):
        backbone = models.resnet101(pretrained = pretrained)
   
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    
    in_channels_stage2 = backbone.inplanes // 8

    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

def mobilenet_fpn_backbone(backbone_name, pretrained):
    backbone = models.mobilenet_v2(pretrained = pretrained).features
    # for name, parameter in backbone.named_parameters():
    #     print(name)
    return_layers = {'15': '0', '16':'1', '17':'2','18': '3'}
    in_channels_list = [
        160, 160, 320, 1280
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

def efficientnet_fpn_backbone(backbone_name, pretrained):
    backbone = EfficientNet.from_pretrained(backbone_name).features
    for name, parameter in backbone.parameters():
        print(name)
    ipdb.set_trace()

def get_fpn_backbone(backbone_name, pretrained):
    if "mobilenet" in backbone_name:
        return mobilenet_fpn_backbone(backbone_name, pretrained)
    elif "resnet" in backbone_name:
        return resnet_fpn_backbone(backbone_name, pretrained)
    elif "efficientnet" in backbone_name:
        return efficientnet_fpn_backbone(backbone_name, pretrained)
    # return efficientnet_fpn_backbone('efficientnet-b0', pretrained)
    # return resnet_fpn_backbone("resnet50", pretrained)