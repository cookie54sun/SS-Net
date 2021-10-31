# Pytorch code for SS-Net 

This is a pytorch implementation of Straight Sampling Network For Point Cloud Learning (ICIP2021).

## Environment

Code is tested on pytorch 1.1.0 and python 3.6.

## How to use this code

- Prepare data

    ```
    wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip --no-check-certificate
    unzip modelnet40_normal_resampled.zip
    mv modelnet40_normal_resampled data/
    ```

- Train
    
    ```
    python train_cls.py
    ```


## Pretrained model

 Pretrained model for classification is stored in folder `/pre_trained`.


## Citation

Please cite this paper if you want to use it in your work.

```
@INPROCEEDINGS{9506477,  author={Sun, Ran and Chen, Gaojie and Ma, Jie and An, Pei},  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},   title={Straight Sampling Network for Point Cloud Learning},   year={2021},  volume={},  number={},  pages={3088-3092},  doi={10.1109/ICIP42928.2021.9506477}}
```






