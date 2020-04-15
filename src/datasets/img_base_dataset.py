"""Image Base dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as osp
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import ipdb

class ImageBaseDataset(Dataset):
    """Image Base dataset to inherit from"""

    def __init__(self, mode, datalst_pth, data_root, lbl_fname = None, lbl_type = None,
                 seq_len=-1, transform = None, spldata_dir = None):
        """Initialize the dataset

        Args:
            mode: `train`, `val`, or `test` mode
            datalst_pth: a dictionary of paths wrt to different modes
            data_root: root directory of the dataset
            lbl_fname: path to the label file
            lbl_type: type of the label to retrieve
            transform: transform object to apply random data augmentation
        """
        # assert mode.split('/')[-1] in self.get_support_modes(), 'Unsupported mode: {}'.format(mode)
        self.mode = mode

        self.datalst_pth = datalst_pth[mode]
        assert osp.isfile(self.datalst_pth), 'Not found: {}'.format(self.datalst_pth)

        self.data_root = osp.join(data_root, spldata_dir[self.mode])
        assert osp.isdir(data_root), 'Not found: {}'.format(data_root)

        #First step: build list of all images in dataset
        self.imglst = self.get_img_list()
        self.preprocess_imglst()
        # Get all imgs' file name
        self.get_all_img_fname()
                
        self.lbl_fname = lbl_fname
        
        # Get all imgs's label
        self.get_all_labels()
        
        # Set up transforms object (#TODO: check again when transforms are neccessary)
        self.transforms = transform
   
    def get_data_sample(self, idx):
        """This function returns a data sample"""
        image = Image.open(self.img_fname_lst[idx])
        return image
    
    def get_img_list(self):
        """This function opens datalst_pth and read everything in that file"""
        return open(self.datalst_pth, 'r').read().splitlines()

    def get_data_label(self, idx):
        """This function returns a label for each image"""
        # Test mode does not have label
        if (self.mode == 'test'):
            return -1
        return self.label_list[idx]

    def get_all_img_fname(self):
        """This function constructs a list of image names.
        Called: self.img_fname_lst 
        Each element should looks like: `/path/to/0000123.jpg'"""
        self.img_fname_lst = [osp.join(self.data_root, img_name) \
                                for img_name in self.imglst]

    def get_all_labels(self):
        """This function constructs a dictionary of image's label. 
        Each element should looks like: 
        lbl_dict[str(`0000123')] = 1 """
        df = pd.read_csv(self.lbl_fname).to_numpy()
        lbl_dict = {}
        
        for i, img_id in enumerate(df[:,0]):
            lbl_dict[str(img_id)] = int(df[i,1])
        # todo : fix -1 hard code! 
        self.label_list= lbl_dict

    def get_support_modes(self):
        """This funnnction return support modes for the current dataset
        """
        return ['train', 'val', 'test']

    def preprocess_imglst(self):
        """Clean images list entry e.g. "00123.jpg" -> int(123) """
        pass
        
    def get_nclasses(self):
        return 0 #todo: fix this!

    def __len__(self):
        return len(self.imglst)

    def __getitem__(self, idx):
        """Get item wrt a given index
        Args:
            idx: sample index
        Returns:
            imgs: PIL Image object
            lbl: (int) label
        """
        # Retrieve image label wrt to the given index
        lbl = self.get_data_label(idx)

        # Retrieve image data wrt to the given index
        data = self.get_data_sample(idx)
        
        # If transform is no none, apply data augmentation strategy
        if self.transforms:
            data = self.transforms(data)     

        return data, lbl

    