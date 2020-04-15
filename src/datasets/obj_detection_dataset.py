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
from src.datasets.img_base_dataset import ImageBaseDataset
import torch

class ObjectDetectionDataset(ImageBaseDataset):
    """The Luna16 Dataset"""
    def get_all_img_fname(self):
        """This function constructs a list of image names.
        Called: self.img_fname_lst 
        Each element should looks like: `0000123.jpg'"""
        self.img_fname_lst = self.imglst.copy()

    def get_all_labels(self):
        df = pd.read_csv(self.lbl_fname)
        lbl_list = []
        for i, seriesuid in enumerate(df['img_name']):
            lbl_list.append(( df['x1'][i], 
                            df['y1'][i], 
                            df['x2'][i],
                            df['y2'][i]))
        return lbl_list

    def get_data_label(self, idx):
        """This function returns a label for each image"""
        pass
        # if (self.mode == 'test'):
            # return -1
        # return self.label_list[idx]
    
    def __getitem__(self, idx):
        """Get item wrt a given index
        Args:
            idx: sample index
        Returns:
            imgs: PIL Image object
            lbl:  standard COCO bboxes
        """
        # Retrieve image label wrt to the given index
        img   = self.get_data_sample(idx).convert('RGB')
        # Retrieve image label wrt to the given index
        boxes, labels = self.get_data_label(idx)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels= torch.as_tensor(labels, dtype=torch.int64)
        #dummy mask:
        masks = np.zeros((1,512,512))
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target