# Dual Regression Compression for DRN

---

![](../figures/DRN.png)

---

## Contents
  - [Contents](#Contents)
  - [Dependencies](#Dependencies)
  - [Datasets](#Datasets)
  - [Models](#Models)


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

You can download the pre-trained large SR models enhanced by our Dual Regression (DR) scheme for 4x and 8x SR. 

|  Method | Scale | Params | FLOPs (G) | Dataset | PSNR (dB) |  SSIM |   Model Zoo  |
|:-------:|:-----:|:------:|:---------:|:-------:|:---------:|:-----:|:------------:|
|  DRN-L  |   4   |  9.8M  |   224.8   |   Set5  |   32.74   | 0.902 | [Download](https://github.com/guoyongcs/DRN/releases/download/v0.1/DRNL4x.pt) |
|  DRN-S  |   4   |  4.8M  |   109.9   |   Set5  |   32.68   | 0.901 | [Download](https://github.com/guoyongcs/DRN/releases/download/v0.1/DRNS4x.pt) |
| DRN-S30 |   4   |  3.1M  |    72.3   |   Set5  |   32.66   | 0.900 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.1/DRNS_4x_Prune_30.pt) |
| DRN-S50 |   4   |  2.3M  |    53.1   |   Set5  |   32.50   | 0.898 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.1/DRNS_4x_Prune_50.pt) |
| DRN-S70 |   4   |  1.4M  |    32.8   |   Set5  |   32.40   | 0.897 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.1/DRNS_4x_Prune_70.pt) |
|  DRN-L  |   8   |  10.0M |   366.5   |   Set5  |   27.43   | 0.792 | [Download](https://github.com/guoyongcs/DRN/releases/download/v0.1/DRNL8x.pt) |
|  DRN-S  |   8   |  5.4M  |   198.0   |   Set5  |   27.41   | 0.790 | [Download](https://github.com/guoyongcs/DRN/releases/download/v0.1/DRNS8x.pt) |
| DRN-S30 |   8   |  3.5M  |   124.1   |   Set5  |   27.37   | 0.790 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.1/DRNS_8x_Prune_30.pt) |
| DRN-S50 |   8   |  2.5M  |    94.5   |   Set5  |   27.26   | 0.785 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.1/DRNS_8x_Prune_50.pt) |
| DRN-S70 |   8   |  1.6M  |    53.6   |   Set5  |   27.16   | 0.783 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.1/DRNS_8x_Prune_70.pt) |


## Evaluating Pre-trained Models


You can use the following command to test our DRN-S and DRN-L models for 4x SR.  

- To test the models for 8x SR, set `--scale` to 8, and change `--pre_train` to the path of the coressponding pretrained model.
- To test the compressed DRN-S model, you may need to set the path of the corresponding channel configuration `--config_path` in the following commands.  


```shell
# Test DRN-S
python main.py --data_dir datasets/ \
--save ../experiments/drns_x4 \
--data_test Set5 --scale 4 --template DRNS \ 
--test_only --save_results \
--pre_train ../pretrained_models/DRNS4x.pt

# Test DRN-L
python main.py --data_dir datasets/ \
--save ../experiments/drnl_x4 \
--data_test Set5 --scale 4 --template DRNL \ 
--test_only --save_results \
--pre_train ../pretrained_models/DRNL4x.pt

# Test DRN-S30 (which removes about 30% parameters from DRN-S) for 4x SR
python main.py --data_dir datasets/ \
--save ../experiments/drns30_x4 \
--data_test Set5 --scale 4 --model SDRN \
--n_resblocks 30 --n_feats 16 \
--test_only --save_results \
--pre_train ../pretrained_models/DRNS_4x_Prune_30.pt \ 
--configs_path ./pretrained_models/search_x4p03.txt

# Test DRN-S30 (which removes about 30% parameters from DRN-S) for 8x SR
python main.py --data_dir datasets/ \
--save ../experiments/drns30_x8 \
--data_test Set5 --scale 8 --model SDRN \
--n_resblocks 30 --n_feats 8 \
--test_only --save_results \
--pre_train ../pretrained_models/DRNS_8x_Prune_30.pt \ 
--configs_path ./pretrained_models/search_x8p03.txt
```




## Training 
First, please download the training and testing datasets, and place them in any folder, such as `datasets/`. 

- You can use the following command to train the DRN-S or DRN-L model for 4x SR. More details can refer to [here](https://github.com/guoyongcs/DRN).

```shell
# train DRN-S
python main.py --data_dir datasets/ --data_train DF2K \
--scale 4 --template DRNS --save ../experiments/DRNS_x4

# train DRN-L
python main.py --data_dir datasets/ --data_train DF2K \
--scale 4 --template DRNL --save ../experiments/DRNL_x4
```

## Model Compression

In our DRC scheme, we first search for a promising channel configuration. Based on the searched channel configuration, we further prune the pre-trained SR model to obtain the compressed model.

- You can use the following command to **search** for a promising channel configuration for removing 30% parameters from DRN-S.  
**Note:** If you want to remove more paramters with different ratio, such as 50% or 70%, simply set the option `--pruning_rate` as 0.5 or 0.7.

```bash
python main.py --search --scale 4 --template DRNS \
--pre_train pretrained_models/DRN/DRNS4x.pt \
--pre_train_dual pretrained_models/DRN/DRNS4x_dual_model.pt \
--epochs 100 --data_dir datasets/ --data_train DIV2KS \
--pruning_rate 0.3 --batch_size 16 --test_every 50 --save ../search_DRNS30/
```

- You can use the following command to **prune** the redundant channels of DRN-S.  
**Note:** You may need to set `--configs_path` to the searched configuration file. For example, you can set `--configs_path ../pretrained_models/DRNS30_4x_config.txt`.

```bash
python main.py --pruning --scale 4 --template DRNS \
--pre_train ../pretrained_models/DRN/DRNS4x.pt \
--pre_train_dual ../pretrained_models/DRN/DRNS4x_dual_model.pt \
--data_dir datasets/ --data_train DIV2KS \
--batch_size 16 --test_every 10 --save ../pruning_DRNS30/ \
--configs_path ../pretrained_models/DRNS30_4x_config.txt
```

- You can use the following command to **finetune** the compressed model.  
**Note 1:** You may need to set `--pruned_model` to the path of the compressed DRN-S model. For example, you can set `--pruned_model ../pruning_DRNS30/model/pruned_model_cs_064.pth`.  
**Note 2:** You need to load the pretrained DRNL model during the finetune process using options `--pre_train`.

```bash
python main.py --finetune --scale 4 --n_GPUs 4 \
--data_dir datasets/ --data_train DF2K \
--template DRNL --ft_model SDRN \
--configs_path ../pretrained_models/DRN/DRNS30_4x_config.txt \
--save ../prune_experiment/finetune_drns30_x4 \
--pre_train ../pretrained_models/DRN/DRNL4x.pt \
--pruned_model ../pruning_DRNS30/model/pruned_model_cs_064.pth \
--pruned_dual_model ../pretrained_models/DRNS4x_dual_model.pt 
```

