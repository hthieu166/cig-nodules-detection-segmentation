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
import pydicom as dcm
import torch
from skimage.transform import resize
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import ipdb
from src.datasets.img_base_dataset import ImageBaseDataset

class DCMSlices(ImageBaseDataset):
    """The Luna16 Dataset"""
    def get_img_list(self):
        return os.listdir(self.data_root)


    def get_all_img_fname(self):
        return 

    def preprocess_imglst(self):
        raw_paths = glob.glob(osp.join(self.data_root,"*"))
        raw_paths = {
            osp.basename(path): path for path in raw_paths
        }
        self.img_fname_lst = [
            raw_paths[img] for img in self.imglst
        ]

        fname_lst = []
        slicenum = []
        # load image pixel and slice location
        for fname in self.img_fname_lst:
            dataset = dcm.filereader.dcmread(fname)
            slicenum.append(float(dataset.get('SliceLocation')))
            fname_lst.append(fname)
        # ordering slices
        fname_lst = np.array(fname_lst)
        slicenum = np.array(slicenum)
        inds = slicenum.argsort()
        self.img_fname_lst = fname_lst[inds]
        
    def get_all_labels(self):
        lbl_dict = {}
        # ipdb.set_trace()
        lbl_dict[self.img_fname_lst[0]] = [0, 0, 0, 0]
        self.lbl_dict = lbl_dict


    def get_data_label(self, idx):
        """This function returns a label for each image
        Returns:
            boxes: list of all boxes in one image each element is [x1,y1,x2,y2]
            labels: list of corressponding labels 
        """
        #Return 1: only 1 label: "nodule"
        labels = list([1])
        boxes = [[0, 0, 0, 0]] 
        return boxes, labels

    # def load_scan(self, path):
    #     freader = dcm.filereader
    #     if (os.path.isdir(path)):
    #         slices = [freader.dcmread(path + '/' + s) for s in os.listdir(path)]
    #         slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    #         try:
    #             slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    #         except:
    #             slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
                
    #         for s in slices:
    #             s.SliceThickness = slice_thickness
    #     else:
    #         slices = [freader.dcmread(path)]
    #     return slices

    # def get_pixels_hu(self, slices):
    #     image = np.stack([s.pixel_array for s in slices])
    #     # Convert to int16 (from sometimes int16), 
    #     # should be possible as values should always be low enough (<32k)
    #     image = image.astype(np.int16)
    #     # Set outside-of-scan pixels to 0
    #     # The intercept is usually -1024, so air is approximately 0
    #     image[image == -2000] = 0
        
    #     # Convert to Hounsfield units (HU)
    #     for slice_number in range(len(slices)):
            
    #         intercept = slices[slice_number].RescaleIntercept
    #         slope = slices[slice_number].RescaleSlope
            
    #         if slope != 1:
    #             image[slice_number] = slope * image[slice_number].astype(np.float64)
    #             image[slice_number] = image[slice_number].astype(np.int16)
                
    #         image[slice_number] += np.int16(intercept)
        
    #     return np.array(image, dtype=np.int16)

    def get_data_sample(self, idx):
        dcm_file = self.img_fname_lst[idx]
        # slices = self.load_scan(dcm_file)
        # dcmimage = self.get_pixels_hu(slices)
        ds = dcm.dcmread(dcm_file)
        # Convert to float to avoid overflow or underflow losses.
        image_2d = ds.pixel_array.astype(float)
        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        pil_img = Image.fromarray(image_2d_scaled).convert('RGB')
        return pil_img

    def normalize_sample(self, smpl_npy):
        min_v = smpl_npy.min()
        max_v = smpl_npy.max()
        smpl_npy = (smpl_npy - min_v) / (max_v - min_v)
        return smpl_npy

        
    def __getitem__(self, idx):
        """Get item wrt a given index
        Args:
            idx: sample index
        Returns:
            imgs: PIL Image object
            lbl:  standard COCO bboxes
        """
        # Retrieve image label wrt to the given index
        img = self.get_data_sample(idx)

        # img = self.normalize_sample(img)[0]
        # img = resize(img, (512, 512), anti_aliasing=True)
        # print(img.shape)


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