"""Load YAML config files"""
import sys
import os

import yaml
import src.utils.logging as logging
import ipdb
logger = logging.get_logger(__name__)

def fix_dataset_params(dataset_params,dict_key):
    new_dict = {}
    data_dict = dataset_params[dict_key]
    for group in data_dict.keys():
        for mem in data_dict[group]:
            new_key = group + "/" + mem
            new_dict[new_key] = data_dict[group][mem]
    dataset_params[dict_key] = new_dict

class ConfigLoader:
    @staticmethod
    def _load_yaml_content(fname):
        """Load and check content of a given YAML file name
        Args:
            fname: path to the config file
        Return:
            content: content of the YAML file
        """
        assert os.path.isfile(fname), 'Config file not found: {}'.format(fname)

        with open(fname, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
                sys.exit(-1)
        return content

    @staticmethod
    def load_model_cfg(fname):
        """Load model config from a given YAML file name
        Parse some configuration parameters if needed.
        Args:
            fname: path to the config file
        Return:
            Name of the model
            The model parameters as a dictionary
        """
        cfg = ConfigLoader._load_yaml_content(fname)

        # Convert kernel_size from list to tupple
        # cfg['model_params']['kernel_size'] = tuple(
        #     x for x in cfg['model_params']['kernel_size'])
        return cfg['model_name'], cfg['model_params']

    @staticmethod
    def load_dataset_cfg(fname):
        """Load dataset config from a given YAML file name.
        Parse some configuration parameters if needed.
        Args:
            fname: path to the config file
        Return:
            Name of the dataset
            The dataset parameters as a dictionary
        """
        cfg = ConfigLoader._load_yaml_content(fname)
        
        if 'train_val' in cfg['dataset_params']['datalst_pth']:
            fix_dataset_params(cfg['dataset_params'],
                'datalst_pth')
            fix_dataset_params(cfg['dataset_params'],
                'spldata_dir')
        return cfg['dataset_name'], cfg['dataset_params']

    @staticmethod
    def load_train_cfg(fname):
        """Load training config from a given YAML name.
        Args:
            fname: path to the config file
        Return:
            The training parameters as a dictionary
        """
        cfg = ConfigLoader._load_yaml_content(fname)

        convert_lst = ['init_lr', 'lr_decay', 'weight_decay', 'momentum']
        for key in convert_lst:
            cfg[key] = float(cfg[key])
        return cfg
