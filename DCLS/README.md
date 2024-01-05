# Dual Regression Compression for DCLS

![](../figures/DCLS.png)


## Contents
- [Dual Regression Compression for DCLS](#dual-regression-compression-for-dcls)
  - [Contents](#contents)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Evaluating Pre-trained Models](#evaluating-pre-trained-models)
  - [Training](#training)
  - [Model Compression](#model-compression)


## Dependencies
```shell
Python>=3.8, PyTorch==1.6, numpy opencv-python lmdb pyyaml
```



## Datasets


Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                        Blind Testing Set                        |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | You can evaluate our blind models on [DIV2KRK](https://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip) dataset.|


To transform datasets to binary files for efficient IO, you can run: 
``` bash
python3 codes/scripts/create_lmdb.py
```
Note that you need to modify the file paths by yourself in `create_lmdb.py`.


## Models

You can download the compressed **blind** SR models (remove 30\% parameters) obtained by our DRC approach.

| Method    | Params | FLOPs (G) | Dataset  | PSNR (dB) |  SSIM  |                          Model Zoo                           |    
| :-------- | :----: | :-------: | :------: | :-------: | :----: | :----------------------------------------------------------: |
| DCLS-DRC | 14.2M |  57.1   | DIV2KRK |  29.01 | 0.798 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/DCLSx4_setting2_DRC.pth) |


## Evaluating Pre-trained Models

You can use the following command the test the DCLS-DRC Model. Note that using a new PyTorch version (later than 1.6) would yield wrong results.  
**Note:** You need to modify the option file to set the paths of dataset, pretrained DCLS-DRC model and the search channel configuration file.

```shell
cd codes/config/DCLS/
# Test DCLS-DRC
python test.py -opt options/setting2/test/test_setting2_x4_drc.yml
```


<!-- ## Training 
First, please download the training and testing datasets, and place them in any folder, such as `datasets/`. 


You can use the following command to train the DCLS model for 4x SR. More details about the training code of DCLS refer to [here](https://github.com/megvii-research/DCLS-SR).  
**Note 1:** Note that using a new PyTorch version (later than 1.6) would yield wrong results.  
**Note 2:** Please modify the configuration file to set paths of yourself, such as set `dataroot_GT` of training set to `datasets/DF2K/DF2K_HR`.

```bash
cd codes/config/DCLS/
python3 -m torch.distributed.launch --nproc_per_node=4 --master_poer=4321 \
train.py -opt=options/setting2/train_setting2_drn_x4.yml --launcher pytorch
``` -->

## Model Compression
In our DRC scheme, we first search for a promising channel configuration. Based on the searched channel configuration, we further prune the pre-trained SR model to obtain the compressed model.

- You can use the following command to **search** for a promising channel configuration for DCLS.   
**Note:** You need to set the option `pretrain_model_G` in the option file to the path of the pretrained DCLS-DR model.  

```bash
cd codes/config/DCLS/
python search.py \
-opt options/setting2/train/search_setting2_x4_drc.yml \
--save_dir ./experiment/DCLS_experiments/search_p03
```

- You can use the following command to **prune** the redundant channels of DCLS.  
**Note:** You need to set the option `configs_path` to the path to the searched channel configuration file for DCLS.  

```bash
cd codes/config/DCLS/
python pruning.py \
-opt options/setting2/train/pruning_setting2_x4_drc.yml \
--save_dir ./experiment/DCLS_experiments/pruning_p03
```

- You can use the following command to **finetune** the compressed model to obtain the final DCLS-DRC.  
**Note 1:** You need to set the option `configs_path` to the path to the searched channel configuration file for DCLS.  
**Note 2:** You need to set the `pretrain_model_G ` in the option file to the path of the pruned DCLS model.

```bash
cd codes/config/DCLS/
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 train.py -opt=options/setting2/train/train_setting2_x4_drc.yml --launcher pytorch --train_dataroot datasets/ --exp_root ./experiment/DCLS
```

