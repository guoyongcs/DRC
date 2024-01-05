# Dual Regression Scheme for DAT 


![](../figures/DAT.png)


## Contents
- [Dual Regression Scheme for DAT](#dual-regression-scheme-for-dat)
  - [Contents](#contents)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Evaluating Pre-trained Models](#evaluating-pre-trained-models)
  - [Training](#training)

## Dependencies
```shell
Python>=3.8, PyTorch>=1.10, numpy, skimage, imageio, matplotlib, tqdm
```

## Datasets

Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                         Non-Blind Testing Set                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | You can evaluate our non-blind models on several widely used [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar), including Set5, Set14, B100, Urban100, Manga109. |



## Models

You can download the pre-trained DAT models enhanced by our Dual Regression (DR) scheme.

| Method    | Params | FLOPs (G) | Dataset  | PSNR (dB) |  SSIM  |                          Model Zoo                           |                       
| :-------- | :----: | :-------: | :------: | :-------: | :----: | :----------------------------------------------------------: |
| DAT-DR    | 14.8M  |   155.1   | Set5 |   33.17  | 0.906  | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/DAT_Dual_Regression.pth) |

## Evaluating Pre-trained Models
You can use the following command the test the DAT-DR Model.  
**Note:** You need to modify the option file to set the paths of dataset and pretrained DAT-DR model. 

```shell
# Test DAT-DR
PYTHONPATH="./:${PYTHONPATH}" python \
basicsr/test.py -opt options/Test/test_DAT_x4_dr.yml
```

## Training 

You can use the following command to train the DAT-DR model for 4x SR.   
**Note:** You may need to change the path of the training and testing dataset in the configuration file. More details about the training code of DAT refer to [here](https://github.com/zhengchen1999/DAT).


```shell
PYTHONPATH="./:${PYTHONPATH}" python \
-m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
basicsr/train.py -opt options/Train/train_DAT_x4_dr.yml \
--launcher pytorch --auto_resume
```