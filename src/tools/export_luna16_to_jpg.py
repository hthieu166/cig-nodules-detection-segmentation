from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys
import os
import os.path as osp
sys.path.insert(0, os.path.abspath('./'))
import numpy as np
from torch.utils.data import DataLoader
from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory, LossFactory, InferFactory
from src.loaders.base_loader_factory import BaseDataLoaderFactory
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils
import src.utils.logging as logging
from PIL import Image, ImageDraw
import pandas as pd
import ipdb

import argparse

def parse_args():
    """Parse input arguments"""
    def str2bool(v):
        """Convert a string to boolean type"""
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o', '--out_dir', type=str,
        help='Path to the output folder')
    parser.add_argument(
        '-d', '--dataset_cfg', type=str,
        help='Path to the dataset config filename')
    parser.add_argument(
        '-b', '--draw_bbox', type=str2bool, default = False,
        help='Do you want to draw a rectangle around nodules?')
    args = parser.parse_args()
    return args

def main():
    data_cfg = args.dataset_cfg
    out_dir  = args.out_dir
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(data_cfg)
    data_fact = DatasetFactory()
    luna16dataset = data_fact.generate(dataset_name, mode="train", **dataset_params)
    luna16loader  = DataLoader(luna16dataset, shuffle=False, drop_last=False) 
    
    pbar = MiscUtils.gen_pbar(max_value=len(luna16loader), msg='Converting: ')
    slc_img_names = []
    img_converted_labels   = []
    
    for i, (samples, labels) in enumerate(luna16loader):
        pbar.update(i+1)
        fname        = luna16dataset.imglst[i]
        smpl_out_dir = osp.join(out_dir, "images", fname)
        smpl = samples[0]
        lbl  = labels
        lbl  = [c.item() for c in lbl]
        x, y, z, rx, ry, rz = tuple(lbl)
    
        smpl_npy   = smpl.squeeze().detach().numpy()
        slices_idx = [max(0, z-rz), min(smpl.shape[0], z+int(rz)) + 1]
        
        
        for slidx in range(*slices_idx):
            smpl_uint  = (smpl_npy[slidx] * 255).astype(np.uint8)
            smpl_pil   = Image.fromarray(np.concatenate([smpl_uint[...,np.newaxis]]*3, axis = 2))
            slc_img_name = str(fname)+"_"+str(slidx)+".jpg"
            slc_img_pth  = osp.join(smpl_out_dir, slc_img_name)
            #Draw bbox
            if (args.draw_bbox):
                smpl_drw = ImageDraw.Draw(smpl_pil)
                smpl_drw.point([y,x], fill='green')
                smpl_drw.rectangle([(x-rx, y-ry),(x+rx,y+ry)], outline = 'green')
            img_converted_labels.append((x-rx, y-ry, x+rx, y+ry))
            slc_img_names.append(slc_img_name)
            os.makedirs(osp.dirname(slc_img_pth), exist_ok=True)
            smpl_pil.save(slc_img_pth)
    
    pbar.finish()
    df = pd.DataFrame(
        {"img_name": slc_img_names,
        "x1": [d[0] for d in img_converted_labels],
        "y1": [d[1] for d in img_converted_labels],
        "x2": [d[2] for d in img_converted_labels],
        "y2": [d[3] for d in img_converted_labels]
        }
    )
    csv_path = osp.join(out_dir,"luna16_nodules_lbl_converted.csv")
    df.to_csv(csv_path, index= False)

if (__name__=="__main__"):
    args = parse_args()
    main()