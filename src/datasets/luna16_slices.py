"""Base dataset for object detection problems"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import glob
import os.path as osp
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import ipdb
from src.datasets.obj_detection_dataset import ObjectDetectionDataset

class Luna16Slices(ObjectDetectionDataset):
    """The Luna16 Dataset"""
    def get_all_img_fname(self):
        raw_paths = glob.glob(osp.join(self.data_root,"*","*.jpg"))
        raw_paths = {
            osp.basename(path): path for path in raw_paths
        }
        self.img_fname_lst = [
            raw_paths[img]
            for img in self.imglst
        ]

    def get_all_labels(self):
        df = pd.read_csv(self.lbl_fname)
        lbl_dict = {}
        # ipdb.set_trace()
        for i, img_name in enumerate(df['img_name']):
            lbl_dict[img_name] = [df['x1'][i], 
                            df['y1'][i], 
                            df['x2'][i],
                            df['y2'][i]]
        self.lbl_dict = lbl_dict

    def get_data_label(self, idx):
        """This function returns a label for each image
        Returns:
            boxes: list of all boxes in one image each element is [x1,y1,x2,y2]
            labels: list of corressponding labels 
        """
        #Return 1: only 1 label: "nodule"
        labels = list([1])
        boxes = [self.lbl_dict[self.imglst[idx]]] 
        return boxes, labels