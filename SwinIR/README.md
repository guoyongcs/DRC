# Dual Regression Compression for SwinIR

![](../figures/SwinIR.png)


## Contents
- [Dual Regression Compression for SwinIR](#dual-regression-compression-for-swinir)
  - [Contents](#contents)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Testing](#testing)
      - [Training](#training)
      - [Model Compression](#model-compression)


## Dependencies
```shell
Python>=3.8, PyTorch>=1.10, numpy, skimage, imageio, matplotlib, tqdm
```


## Datasets

Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                         Non-Blind Testing Set                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | You can evaluate our non-blind models on several widely used [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar), including Set5, Set14, B100, Urban100, Manga109. |


To transform datasets to binary files for efficient IO, you can run: 
``` bash
python scripts/data_preparation/create_lmdb.py
```
**Note:** You may need to modify the file paths by yourself in `create_lmdb.py`.

## Models

You can download the pre-trained large SR models enhanced by our Dual Regression (DR) scheme for 4x SR. And t

| Method    | Params | FLOPs (G) | Dataset  | PSNR (dB) |  SSIM  |                          Model Zoo                           |                       
| :-------- | :----: | :-------: | :------: | :-------: | :----: | :----------------------------------------------------------: |
| SwinIR-DR | 11.9M |  121.1   | Set5 |   33.03 | 0.904 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/SwinIR_Dual_Regression.pth) |
| SwinIR-light-DR | 897K |  10.0   | Set5 |   32.44 | 0.897 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/SwinIR_light_DR.pth) |
| SwinIR-light-DRC     | 635K |  6.8    | Set5 |   32.44 | 0.896 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/SwinIR_light_DRC.pth) |


## Testing 

You can use the following command to test the SwinIR-DR and SwinIR-light-DRC models for 4x SR.      
**Note 1:** To test models, you need to modify the option file to set the paths of yourself, such as the path of the pretrained models `pretrained_netG`.
**Note 2:** To test SwinIR-light-DR, you need to modify the option file to set `configs_path` to the path of the searched channel configuration for SwinIR-light model.

```shell
# Test SwinIR-DR
python main_test_swinir_dr.py --task classical_sr_dr \
--opt options/swinir/test_swinir_sr_classical_dr.json \
--folder_lq /data_root/dataset/sr/benchmark/Set5/LR_bicubic/X4 \
--folder_gt /data_root/dataset/sr/benchmark/Set5/HR --dataset Set5

# Test SwinIR-light-DRC
python main_test_swinir_drc.py --task lightweight_sr_drc \
--opt options/swinir/test_swinir_sr_lightweight_drc.json \
--folder_lq /data_root/dataset/sr/benchmark/Set5/LR_bicubic/X4 \
--folder_gt /data_root/dataset/sr/benchmark/Set5/HR --dataset Set5
```

#### Training

You can use the following command to train the SwinIR-DR model for 4x SR. Note that the path of the training and testing dataset in the configuration file may need to be changed. More details about the training code of SwinIR refer to [here](https://github.com/cszn/KAIR).

```shell
# train SwinIR-DR
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 \
main_train_psnr.py --opt options/swinir/train_swinir_sr_classical_dr.json --dist True

# train SwinIR-light-DR
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 \
main_train_psnr.py --opt options/swinir/train_swinir_sr_lightweight_dr.json --dist True
```



#### Model Compression

In our DRC scheme, we first search for a promising channel configuration. Based on the searched channel configuration, we further prune the pre-trained SR model to obtain the compressed model.

- You can use the following command to **search** for a promising channel configuration for SwinIR-light.   
**Note:** You need to set the `pretrained_netG` in the option file to the path of the pretrained SwinIR-light-DR model.


```bash
python main_search_swinir.py \
--opt_path options/swinir/search_swinir_sr_lightweight_drc.json \
--save_dir ../experiments/search_swinir_sr_lightweight_drc
```

- You can use the following command to **prune** the redundant channels of SwinIR-light.  
**Note 1:** You need to set the `pretrained_netG` in the option file to the path of the pretrained SwinIR-light-DR model.   
**Note 2:** You need to set the `configs_path` in the option file to the searched channel configuration for SwinIR-light model.  

```bash
python main_pruning_swinir.py \
--opt_path options/swinir/pruning_swinir_sr_lightweight_drc.json \
--save_dir ../experiments/pruning_swinir_sr_lightweight_drc
```

- You can use the following command to **finetune** the compressed model to obtain the final SwinIR-light-DRC.   
**Note 1:** You need to set the `configs_path` in the option file to the searched channel configuration for SwinIR-light.   
**Note 2:** You need to set the `pretrained_netG` in the option file to the path of the pruned SwinIR-light model.  
**Note 3:** You need to set the `pretrained_netG` in the teacher option file to the path of the pretrained SwinIR model.


```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1235 \
main_train_psnr_drc.py --opt options/swinir/train_swinir_sr_lightweight_drc.json \ 
--opt_teacher options/swinir/train_swinir_sr_classical.json --dist True
```