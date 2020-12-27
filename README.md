# kaggle-aptos2019-blindness-detection
##  Train
Use following commands to run ```train.pyˋˋˋ
ˋˋˋ
python train.py --arch se_resnext50_32x4d
python train.py --arch se_resnext101_32x4d --batch_size 24
python train.py --arch senet15
ˋˋˋ


### 1st-level models (run on local)
- Models: SE-ResNeXt50_32x4d, SE-ResNeXt101_32x4d, SENet154
- Loss: MSE
- Optimizer: SGD (momentum=0.9)
- LR scheduler: CosineAnnealingLR (lr=1e-3 -> 1e-5)
- 30 epochs
- Dataset: 2019 train dataset (5-folds cv) + 2015 dataset
### 2nd-level models (run on kernel)
- Models: SE-ResNeXt50_32x4d, SE-ResNeXt101_32x4d (1st-level models' weights)
- Loss: MSE
- Optimizer: RAdam
- LR scheduler: CosineAnnealingLR (lr=1e-3 -> 1e-5)
- 10 epochs
- Dataset: 2019 train dataset (5-folds cv) + 2019 test dataset
- Pseudo labels: weighted average of 1st-level models
