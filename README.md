# Dual Regression Compression (DRC)

[Yong Guo](http://www.guoyongcs.com/), [Jingdong Wang](https://jingdongwang2017.github.io/), [Qi Chen](https://scholar.google.com/citations?hl=zh-CN&user=OgKU77kAAAAJ&view_op=list_works&sortby=pubdate), [Jiezhang Cao](https://www.jiezhangcao.com/), [Zeshuai Deng](https://scholar.google.com/citations?hl=zh-CN&user=udPURMAAAAAJ), [Yanwu Xu](https://scholar.google.com/citations?hl=zh-CN&user=0jP8f7sAAAAJ), Jian Chen, [Mingkui Tan](https://tanmingkui.github.io/)


This repository contains the official Pytorch implementation and the pretrained models of [Towards Lightweight Super-Resolution with Dual Regression Learning](https://arxiv.org/pdf/2207.07929.pdf).



---
![](figures/DRC.png)



## Contents
- [Dual Regression Compression (DRC)](#dual-regression-compression-drc)
  - [Contents](#contents)
  - [Catalog](#catalog)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Evaluating and Training](#evaluating-and-training)
  - [Results](#results)
  - [Citation](#citation)



## Catalog
- [x] We release the pre-trained models of some large SR models enhanced by our Dual Regression (DR).
- [x] We release the pre-trained models of the non-blind SR models compressed by our Dual Regression Compression (DRC).
- [x] We release the pre-trained models of the blind SR models compressed by our Dual Regression Compression (DRC).
- [x] We release the code of evaluation and training.



## Datasets


Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                         Non-Blind Testing Set                          |                        Blind Testing Set                        |
| :----------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | You can evaluate our non-blind models on several widely used [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar), including Set5, Set14, B100, Urban100, Manga109. | You can evaluate our blind models on [DIV2KRK](https://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip) dataset.|

Please organize the datasets using the following hierarchy.
```
- datasets/
    - DIV2K
        - DIV2K_train_HR
        - DIV2K_train_LR_bicubic

    - DF2K
        - DF2K_HR
        - DF2K_LR_bicubic
    
    - benchmark
        - Set5
        - Set14
        - B100
        - Urban100
        - Manga109
    
    - DIV2KRK
        - gt
        - lr_x2 
        - lr_x4
```



## Models

You can download the pre-trained large SR models enhanced by our Dual Regression (DR) scheme for 4x SR.
More pretrained models can be found in the released assets of this repository.

| Method    | Params | FLOPs (G) | Dataset  | PSNR (dB) |  SSIM  |                          Model Zoo                           |                       
| :-------- | :----: | :-------: | :------: | :-------: | :----: | :----------------------------------------------------------: |
| DRN-S     | 4.8M |  109.9    | Set5 |   32.68 | 0.901 | [Download](https://github.com/guoyongcs/DRN/releases/download/v0.1/DRNS4x.pt) |
| DRN-L     | 9.8M |  224.8    | Set5 |   32.74 | 0.902 | [Download](https://github.com/guoyongcs/DRN/releases/download/v0.1/DRNL4x.pt) |
| SwinIR-DR | 11.9M |  121.1   | Set5 |   33.03 | 0.904 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/SwinIR_Dual_Regression.pth) |
| DAT-DR    | 14.8M  |   155.1   | Set5 |   33.17  | 0.906  | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/DAT_Dual_Regression.pth) |

You can download the compressed **non-blind** SR models (remove 30\% parameters) obtained by our Dual Regression Compression (DRC) approach for 4x SR.

| Method    | Params | FLOPs (G) | Dataset  | PSNR (dB) |  SSIM  |                          Model Zoo                           |                       
| :-------- | :----: | :-------: | :------: | :-------: | :----: | :----------------------------------------------------------: |
| DRN-S30     | 3.1M |  72.3    | Set5 |   32.66 | 0.900 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.1/DRNS_4x_Prune_30.pt) |
| SwinIR-light-DRC     | 635K |  6.8    | Set5 |   32.44 | 0.896 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/SwinIR_light_DRC.pth) |

You can download the compressed **blind** SR models (remove 30\% parameters) obtained by our DRC approach for 4x SR.

| Method    | Params | FLOPs (G) | Dataset  | PSNR (dB) |  SSIM  |                          Model Zoo                           |    
| :-------- | :----: | :-------: | :------: | :-------: | :----: | :----------------------------------------------------------: |
| DCLS-DRC | 14.2M |  57.1   | DIV2KRK |  29.01 | 0.798 | [Download](https://github.com/guoyongcs/DRC/releases/download/v1.0/DCLSx4_setting2_DRC.pth) |

## Evaluating and Training

We put the detailed explanations about the code of evaluating and training in the corresponding folders. Please refer to more details in the `README.md` file within these folders.






## Results

We achieved competitive performance. Detailed results can be found in the paper.

<details>
<summary>Click to expand</summary>

- Comparison results with SOTA SR methods for 4x SR in Table 1.

<p align="center">
  <img width="900" src="figures/Table-1.png">
</p>


- Comparison results with lightweight SR methods for 4x SR in Table 2.

<p align="center">
  <img width="900" src="figures/Table-2.png">
</p>

- Comparison results with blind SR methods for 4x SR in Table 3

<p align="center">
  <img width="450" src="figures/Table-3.png">
</p>


- Visual comparison (x4) with SOTA methods for 4x SR.

<p align="center">
  <img width="900" src="figures/img4x_compare.jpg">
</p>

- Visual comparison (x4) with compression methods for 4x SR.

<p align="center">
  <img width="900" src="figures/img4x_compression_compare.jpg">
</p>
</details>

---


## Citation
If you find this repository helpful, please consider citing:
```
@article{guo2022towards,
  title={Towards lightweight super-resolution with dual regression learning},
  author={Guo, Yong and Wang, Jingdong and Chen, Qi and Cao, Jiezhang and Deng, Zeshuai and Xu, Yanwu and Chen, Jian and Tan, Mingkui},
  journal={arXiv preprint arXiv:2207.07929},
  year={2022}
}
```