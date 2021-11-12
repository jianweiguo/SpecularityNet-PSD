# SpecularityNet-TMM

This is the project of TMM 2021 paper "Single-Image Specular Highlight Removal via Real-World Dataset Construction"

Specular reflections pose great challenges on various multimedia and computer vision tasks. In this paper, we build a large-scale Paired Specular-Diffuse (PSD) image dataset, where the images are carefully captured by using real-world objects and the ground-truth specular-free diffuse images are provided. To the best of our knowledge, this is the first real-world benchmark dataset for specular highlight removal task, which is useful for evaluating and encouraging new deep learning-based approaches. Given this dataset, we present a novel Generative Adversarial Network (GAN) for specular highlight removal from a single image by introducing the detection of specular reflection information as a guidance. Our network also makes full use of the attention mechanism and is able to directly model the mapping relation between the diffuse area and the specular highlight area without any explicit estimation of the illumination. 

<img src="specularity.png" alt="specularity" style="zoom: 50%;" />

## PSD dataset (Paired Specular-Diffuse image dataset)

We collect a total of 13,380 images captured on 2,210 different scenes. We use different objects and backgrounds to build our dataset.  

The dataset can be divided into two different polarization conditions. One consists of 1010 groups of images photographed with fixed polarization angles, and the other one consists of 1200 pairs of images photographed with random polarization angles. For the former,  each group contains 12 images photographed with 12 fixed polarization angles.

![2](dataset_example.png)

**Our full dataset is publicly available**.   

The dataset download link is: https://drive.google.com/file/d/1PUrUyUd1ys6pue8jPGchCGVdFBAf8eqg/view?usp=sharing


## Requisites
* Python =3.7, PyTorch = 1.7.0
* Platforms: Ubuntu 16.04, cuda-10.0


## Quick Start
#### Training dataset
* Modify gpu id, dataset path, and checkpoint path. Adjusting some other parameters if you like.
  
* Please run the following code: 

  ```
  CUDA_VISIBLE_DEVICES=1 python train_specularitynet.py --name refined --inet refined --iters 1 --suffix iters1 --enhance de --freq 0.25 --noise True --lambda_coarse 0.5 --lambda_detect 1.0 --batchSize 4 --nThreads 24 --fliplr 0.5 --flipud 0.5
  ```

#### Testing dataset
* Modify test dataset path and result path.
* Please run the following code: 

```
CUDA_VISIBLE_DEVICES=0 python test_specularitynet.py -r --name refined --inet refined --iters 1 --suffix iters1 --enhance de --batchSize 16 --nThreads 32
```



## Citation

If you find our code helpful in your research or work please cite our paper.

```
@article{wu2021single,
  title={Single-Image Specular Highlight Removal via Real-World Dataset Construction},
  author={Wu, Zhongqi and Zhuang, Chuanqing and Shi, Jian and Guo, Jianwei and Xiao, Jun and Zhang, Xiaopeng and Yan, Dong-Ming},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```

