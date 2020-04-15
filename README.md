# CIG - Nodules Detection and Segmentation on LUNA16
------
Faster R-CNN and Mask R-CNN for nodules detection and segmentation on LUNA 16 Dataset 
## Installation
----
### Environment & Dependencies
A conda environment file info `environment.yml`, and dependencies `requirements.txt` are given, to install all requires packages:
```
$ conda env -f environment.yml
$ conda activate py3torch
$ pip install -r requirements.txt
```
### Dataset
LUNA 3D Volumes should be sliced into 2D Images (Slices do not contain any nodule are filtered out). You can download via [Google Drive](https://drive.google.com/file/d/10Hqd3uAAGAGcVJrvJyzVzSyqy0wYwOFJ/view?usp=sharing) or convert the dataset again. Suppose the original LUNA 16 Dataset is organized as:
```
luna16/
    subsets/
        subset0/
        subset1/
        ...
        subset9/
    annotations.csv
```
Modify the paths in the following file, with new directory (if necessary):
```
configs/dataset_cfgs/luna16.yaml
```
To convert the dataset again, run:
```
$ ./scripts/luna16_export_to_jpg.sh
```
Modify the paths in the following file, with new directory (if necessary):
```
configs/dataset_cfgs/luna16_slices.yaml
```

### Training & Testing
------
For your convenience, the default `GPU_ID` and `N_WORKERS` can be assigned under `scripts/master_env.sh` 
**To train:**
```
$ ./scripts/luna16_frcnn_train.sh ${GPU_ID}
```
**To evaluate:**
```
$ ./scripts/luna16_frcnn_eval.sh ${GPU_ID}
```
Results are given in the `logs/` folder, they includes the checkpoints, and stdout. Prediction results are givien as a pickle file under `outputs/luna16_frcnn/`

Evaluation report for epoch 9, usinng COCO Eval API should looks like this:
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.523
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.117
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.363
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.351
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000

```
#### Tools & Visualization
----
Tools for spliting training-validation sets, and prediction visualization are given in
`LUNA16_tools.ipynb`

#### Notes:
----
If you want to modify the codes, create your new banch and do not push it on master.
If you want to use your own model, loss, sampling objects, data augmentation strategy, etc. You can implement them inside the corrresponding folders inside `src/`. Remember to follow the interfaces and registered them under:
```
src/factories.py
```
### Acknowledgement
The template for this repo was first initialized from a repo of `knmac`.
The implement is heavily borrowed from this [Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)