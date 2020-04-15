"""Training routine"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import time
import ipdb
from torch.utils.tensorboard import SummaryWriter

from tester import test
from src.utils.misc import MiscUtils
import src.utils.logging as logging
from src.models.detectors.detection.engine import train_one_epoch, evaluate
logger = logging.get_logger(__name__)


def train(model, optimizer, criterion, loaders, logdir,
          train_mode, train_params, device, pretrained_model_path, infer_fn):
    """Training routine

    Args:
        model: model to train
        optimizer: optimizer to optimize the loss function
        criterion: loss function
        loaders: dictionary of data loader for training and validation
            'loaders[train]' for training
            'loaders[val]' for validationn
            ...
        logdir: where to store the logs
        train_mode: how to start the training. Only accept values as:
            'from_scratch': start the training from scratch
            'from_pretrained': start the training with a pretrained model
            'resume': resume an interrupted training
        train_params: training parameters as a dictionary
        device: id of the device for torch to allocate objects
        infer_fn: BaseInference object: calculate additional metrics, saving predictions 
    """
    # Setup training starting point
    if train_mode == 'from_scratch':
        start_epoch = 0
        lr = train_params['init_lr']
    elif train_mode == 'from_pretrained':
        model.load_model(pretrained_model_path)
        start_epoch = 0
        lr = train_params['init_lr']
    elif train_mode == 'resume':
        prefix = MiscUtils.get_lastest_checkpoint(logdir)
        lr, start_epoch = MiscUtils.load_progress(model, optimizer, prefix)
    else:
        raise ValueError('Unsupported train_mode: {}'.format(train_mode))

    # Set up some variables
    infer_fn.init_best_model_score()

    # writer = SummaryWriter(log_dir=logdir, purge_step=start_epoch)
    writer = SummaryWriter(log_dir=logdir)

    # Go through training epochs
    for epoch in range(start_epoch, train_params['n_epochs']):
        logger.info('epoch: %d/%d' % (epoch+1, train_params['n_epochs']))

        # Training phase
        train_loader = loaders['train']
        run_iter = epoch * len(train_loader)
        
        #train_one_epoch from torchvision.detection:
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        
        
        # Validation phase
        # val_loss, val_score = test(model, criterion, loaders, device, infer_fn)

        # t = evaluate(model, val_loader, device=device)
        
        # Log using Tensorboard
        
        # writer.add_scalars('losses', {'train': train_loss, 'val': val_loss}, epoch)
        # writer.add_scalar('val_score', val_score, epoch)

        # Save training results when necessary
        if (epoch+1) % train_params['n_epochs_to_log'] == 0:
            MiscUtils.save_progress(model, optimizer, logdir, epoch)

        # Backup the best model
        # if infer_fn.is_better_model(val_loss, val_score):
        #     model.save_model(os.path.join(logdir, 'best.model'))

        # Decay learning Rate
        if epoch+1 in train_params['decay_epochs']:
            lr *= train_params['lr_decay']
            logger.info('Change learning rate to: %.5f' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                logger.info('%.5f' % param_group['lr'])

        logger.info('-'*80)

    writer.close()