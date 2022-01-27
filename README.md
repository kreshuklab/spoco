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
This implementation uses `DistributedDataParallel` training/prediction. In order to restrict the number of GPUs used for training
use `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=0 python spoco_train.py ...` will execute training on `GPU:0`.

### CVPPP dataset 
We used A1 subset of the [CVPPP2017_LSC challenge](https://competitions.codalab.org/competitions/18405) for training. In order to train with 10% of randomly selected objects, run:
```bash
python spoco_train.py \
    --spoco
    --cos
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
    --log-after-iters 256  --max-num-iterations 80000 
```

`CVPPP_ROOT_DIR` is assumed to have the following subdirectories:
```
- train:
    - A1:
        - plantXXX_rgb.png
        - plantXXX_label.png
        ...
    - ...
    - A4:
        - ...
- val:
    - A1:
        - plantXXX_rgb.png
        - plantXXX_label.png
        ...
    - ...
    - A4:
        - ...

```
Since the CVPPP dataset consist of only `training` and `testing` subdirectories, one has to create the train/val split manually using the `training` subdir.

### 3D Training

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
Results will be saved in the given `OUTPUT_DIR` directory. For each test input image `plantXXX_rgb.png` the following
3 output files will be saved in the `OUTPUT_DIR`:
* `plantXXX_rgb_predictions.h5` - HDF5 file with datasets `/raw` (input image), `/embeddings1` (output from the `f` embedding network), `/embeddings2` (output from the `g` momentum contrast network)
* `plantXXX_rgb_predictions_1.png` - output from the `f` embedding network PCA-projected into the RGB-space
* `plantXXX_rgb_predictions_2.png` - output from the `g` momentum contrast network PCA-projected into the RGB-space


## Clustering
To produce the final segmentation one needs to cluster the embeddings with and algorithm of choice. Supported
algoritms: mean-shift, HDBSCAN and Consistency Clustering (as described in the paper). E.g. to cluster CVPPP with HDBSCAN, run:
```bash
python cluster_predictions.py --ds-name cvppp \
    --emb-dir PREDICTION_DIR \
    --output-dataset hdbscan_seg
    --clustering hdbscan --delta-var 0.5 --min-size 200 --remove-largest
```

Where `PREDICTION_DIR` is the directory where h5 files containing network predictions are stored. Resulting segmentation
will be saved as a separate dataset (named `hdbscan_seg` in this example) inside each of the H5 prediction files.