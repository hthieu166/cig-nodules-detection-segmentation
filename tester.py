"""Validation/Testing routine"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch

from src.utils.misc import MiscUtils
from src.inferences.base_infer import BaseInfer
import src.utils.logging as logging
logger = logging.get_logger(__name__)
import os
import os.path as osp
import ipdb
from src.models.detectors.detection.engine import train_one_epoch, evaluate
import pickle as pkl

def test(model, criterion, loaders, device, infer_fn):
    """Evaluate the performance of a model

    Args:
        model: model to evaluate
        criterion: loss function
        loader: dictionary of data loaders for testing
        device: id of the device for torch to allocate objects
        infer_fn: BaseInference object: calculate additional metrics, saving predictions 
    Return:
        test_loss: average loss over the test dataset
        test_score: score over the test dataset
    """
    test_loader = loaders['test']
    _, outputs = evaluate(model, test_loader, device=device)
    
    os.makedirs(infer_fn.output_dir, exist_ok = True)
    file_name = osp.join(infer_fn.output_dir, "test_predictions.pkl")
    
    pkl.dump((outputs, test_loader.dataset.imglst), open(file_name, "bw"))
    return 
