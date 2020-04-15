"""LUNA16 dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import glob
import os.path as osp
import SimpleITK as sitk
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import ipdb
from src.datasets.img_base_dataset import ImageBaseDataset

class Luna16Dataset(ImageBaseDataset):
    """The Luna16 Dataset"""
    def __init__(self, **kwargs):
        self.blacklist = [
        "1.3.6.1.4.1.14519.5.2.1.6279.6001.771741891125176943862272696845"
        ]
        super().__init__(**kwargs)

    def preprocess_imglst(self):
        self.imglst = [img for img in self.imglst \
            if img not in self.blacklist ]
        

    def get_all_img_fname(self):
        raw_paths = glob.glob(osp.join(self.data_root,"*","*.mhd"))
        raw_paths = {
            osp.basename(path).replace(".mhd",""): path for path in raw_paths
        }
        self.img_fname_lst = [
            raw_paths[img] for img in self.imglst
        ]

    def get_all_labels(self):
        df = pd.read_csv(self.lbl_fname)
        lbl_list = []
        for i, seriesuid in enumerate(df['seriesuid']):
            if seriesuid in self.blacklist:
                continue
            lbl_list.append(( df['coordX'][i], 
                            df['coordY'][i], 
                            df['coordZ'][i],
                            df['diameter_mm'][i]))        
        self.label_list = lbl_list

    def get_data_sample(self, idx):
        mhd_file = self.img_fname_lst[idx]
        itkimage = sitk.ReadImage(mhd_file)
        return itkimage

    def normalize_sample(self,smpl_npy):
        min_v = smpl_npy.min()
        max_v = smpl_npy.max()
        smpl_npy = (smpl_npy - min_v) / (max_v - min_v)
        return smpl_npy

    def __getitem__(self, idx):
        itkimg = self.get_data_sample(idx)
        x,y,z,d = self.get_data_label(idx)
        ix,iy,iz = itkimg.TransformPhysicalPointToIndex((x,y,z)) 
        s = np.array(list(reversed(itkimg.GetSpacing())))
        npy_img  = sitk.GetArrayFromImage(itkimg)
        npy_img = self.normalize_sample(npy_img)
        r = d / 2.0
        rx = int(r/s[1])
        ry = int(r/s[2])
        rz = int(r/s[0])
        return npy_img, (ix,iy,iz,rx,ry,rz)

    