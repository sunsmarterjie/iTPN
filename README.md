<div align="center">
<h1>iTPN</h1>
<h3>Integrally Pre-Trained Transformer Pyramid Networks</h3>

Yunjie Tian<sup>1</sup>,
[Lingxi Xie](https://scholar.google.com/citations?user=arny77IAAAAJ&hl=en&oi=ao)<sup>2</sup>, 
Zhaozhi Wang<sup>1</sup>, 
[Longhui Wei](https://scholar.google.com/citations?user=SH_-B_AAAAAJ&hl=en&oi=ao)<sup>2</sup>,
[Xiaopeng Zhang](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en&oi=ao)<sup>2</sup>,
[Jianbin Jiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en&oi=ao)<sup>1</sup>,
[Yaowei Wang](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en&oi=ao)<sup>3</sup>,
[Qi Tian](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en&oi=ao)<sup>2</sup>,
[Qixiang Ye](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en&oi=ao)<sup>1,3</sup>,

<sup>1</sup> [University of Chinese Academy of Sciences], <sup>2</sup> [Huawei Inc.](https://mmlab.ie.cuhk.edu.hk/), <sup>3</sup> [Pengcheng Lab.](https://www.sensetime.com/cn).
  
This repo is the official implementation of [Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/abs/22xx.xxx). 

  
</div>
  
  
> **ImageNet Pretrain**: See [PRETRAIN.md](PRETRAIN.md).\
> **ImageNet Finetune**: See [FINETUNE.md](FINETUNE.md).\
> **Object Detection**: See [DETECTION.md](DET/DETECTION.md).\
> **Semantic Segmentation**: See [SEGMENTATION.md](SEG/SEGMENTATION.md). \


## Getting Started
## requiments
* Ubuntu
* Python 3.7+
* CUDA 10.2+
* GCC 5+
* Pytorch 1.7+
## Dataset
* ImageNet-1K
* COCO2017
* ADE20K

iTPN supports pre-training using pixel and CLIP as supervision. For latter, please first download the [CLIP models](https://github.com/openai/CLIP/blob/main/clip/clip.py)(We use [CLIP-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) and [CLIP-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) models in the paper). See details at [PRETRAIN.md](PRETRAIN.md).


# Due to data privacy policy of Huawei Inc., the checkpoints are not availble temporarily. But we are applying for them and then release them recently.

## Fine-Tuning Results on ImageNet-1K
| Methods | Arch. | Sup. | epochs | Param. (M) | FT acc@1(%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| BEiT | ViT-B | DALLE | 800 | 86 | 83.0 |
| MAE | ViT-B | RGB | 1600 | 86 | 83.6 |
| SimMIM | Swin-B | RGB | 800 | 88 | 84.0 | 
| MaskFeat | ViT-B | HOG | 800 | 86 | 84.0 ||
| data2vec | ViT-B | RGB | 800 | 86 | 84.2 |
| HiViT | HiViT-B | RGB |  1600 | 66 | 84.6 |
| MVP | ViT-B | CLIP-B |  300 | 86 | 84.4 |
| BEiT-v2 | ViT-B | CLIP-B |  1600 | 86 | 85.5 |
| iTPN-B | HiViT-B | RGB | 1600 | 79 | 85.5 |
| iTPN-B | HiViT-B | CLIP-B | 800 | 79 | 86.2 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| iTPN-L | HiViT-L/16 | CLIP-B | 300 | 288 | 87.0 |
| iTPN-L | HiViT-L/16 | CLIP-L | 800 | 288 | 87.8 |
| iTPN-L | HiViT-L/14 | CLIP-L | 300 | 288 | 88.0 |



## License
iTPN is released under the [MIT License](https://github.com/sunsmarterjie/iTPN/blob/main/LICENSE).

## Citation

```bash
@article{tian2022integrally,
  title={Integrally Pre-Trained Transformer Pyramid Networks},
  author={Yunjie, Tian and Lingxi, Xie and Zhaozhi, Wang and Longhui, Wei and Xiaopeng, Zhang and Jianbin, Jiao and Yaowei, Wang and Qi, Tian and Qixiang, Ye},
  journal={arXiv preprint arXiv:22xx.xxxxx},
  year={2022}
}
```
