# SPOCO: Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings

![alt text](./img/Figure2.svg)

This repository provides a PyTorch implementation of our method from the [paper](https://arxiv.org/abs/2103.14572):

```
@misc{wolny2021spoco,
      title={Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings}, 
      author={Adrian Wolny and Qin Yu and Constantin Pape and Anna Kreshuk},
      year={2021},
      eprint={2103.14572},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Installation
Checkout the repo and set up conda environment:
```bash
conda env create -f environment.yaml
```

Activate the new environment:
```bash
conda activate spoco
```

## Training
This implementation uses `DataParallel` training/prediction. In order to restrict the number of GPUs used for training
use `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=0 python spoco_train.py ...` will execute training on `GPU:0`.

### CVPPP dataset 
We used A1 subset of the [CVPPP2017_LSC challenge](https://competitions.codalab.org/competitions/18405) for training. In order to train with 10% of randomly selected objects, run:
```bash
python spoco_train.py \
    --ds-name cvppp --ds-path CVPPP_ROOT_DIR \
    --instance-ratio 0.1 \
    --batch-size 4  \
    --model-name UNet2D \
    --model-layer-order bcr \
    --model-feature-maps 16 32 64 128 256 512 \ 
    --learning-rate 0.0002 \
    --weight-decay 0.00001 \
    --loss-delta-var 0.5 \
    --loss-delta-dist 2.0 \
    --loss-unlabeled-push 1.0 \ 
    --loss-instance-weight 1.0 \
    --loss-consistency-weight 1.0 \
    --kernel-threshold 0.5 \
    --checkpoint-dir CHECKPOINT_DIR \ 
    --log-after-iters 256 --validate-after-iters 512 --max-num-iterations 80000 
```

`CVPPP_ROOT_DIR` is assumed to have the following subdirectories:
```
- training:
    - A1:
        - plantXXX_rgb.png
        - plantXXX_label.png
        ...
    - ...
    - A4:
        - ...
- testing:
    - A1:
        - plantXXX_rgb.png
        - plantXXX_fg.png
        ...
    - ...
    - A4:
        - ...

```

### DSB dataset
We used the data from the [DSB 2018 challenge](https://www.kaggle.com/c/data-science-bowl-2018) randomly split into
train/val/test. In order to train with sparse supervision (10% of randomly selected objects), run:
```bash
python spoco_train.py \
    --ds-name dsb --ds-path DSB_ROOT_DIR \
    --instance-ratio 0.1 \
    --batch-size 4  \
    --model-name UNet2D \
    --model-layer-order gcr \
    --model-feature-maps 16 32 64 128 256 512 \
    --learning-rate 0.0002 \
    --weight-decay 0.00001 \
    --loss-delta-var 0.5 \
    --loss-delta-dist 2.0 \
    --loss-unlabeled-push 1.0 \ 
    --loss-instance-weight 1.0 \
    --loss-consistency-weight 1.0 \
    --kernel-threshold 0.5 \
    --checkpoint-dir CHECKPOINT_DIR \ 
    --log-after-iters 250 --validate-after-iters 500 --max-num-iterations 100000 
```

`DSB_ROOT_DIR` is assumed to have the following subdirectories:
```
- train:
    - images:
        - img1.tif
        - img2.tif
        ...
    - masks:
        - img1.tif
        - img2.tif
        ...
- val:
    - images:
        ...
    - masks:
        ...
- test:
    - images:
        ...
    - masks:
        ...
```

### GAN Training

TODO

## Prediction
Give a model trained on the CVPPP dataset, run the prediction using the following command:
```bash
python spoco_predict.py \
    --ds-name cvppp --ds-path CVPPP_ROOT_DIR --batch-size 4 \ 
    --model-path MODEL_DIR/best_checkpoint.pytorch \
    --model-name UNet2D \
    --model-layer-order bcr \
    --model-feature-maps 16 32 64 128 256 512 \
    --output-dir OUTPUT_DIR
```
Results will be saved in the given `OUTPUT_DIR` directory.

Similarly, to predict on DSB run:
```bash
python spoco_predict.py \
    --ds-name dsb --ds-path DSB_ROOT_DIR --batch-size 4 \ 
    --model-path MODEL_DIR/best_checkpoint.pytorch \
    --model-name UNet2D \
    --model-layer-order gcr \
    --model-feature-maps 16 32 64 128 256 512 \
    --output-dir OUTPUT_DIR
```

## Clustering

TODO