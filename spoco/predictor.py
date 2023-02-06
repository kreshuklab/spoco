import os

import h5py
import numpy as np
import torch
from PIL import Image

from spoco.utils import pca_project


class EmbeddingsPredictor:
    def __init__(self, model, test_loader, output_dir, spoco):
        self.model = model
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.spoco = spoco

    def predict(self):
        # set the model in evaluation mode explicitly
        self.model.eval()

        # run predictions on the entire test_set
        with torch.no_grad():
            for t in self.test_loader:
                if self.spoco:
                    img1, img2, path = t
                    # send batch to device
                    img1, img2 = img1.cuda(), img2.cuda()
                    # forward pass
                    emb1, emb2 = self.model(img1, img2)
                    # iterate over the batch
                    for single_img, single_emb1, single_emb2, single_path in zip(img1, emb1, emb2, path):
                        self.process_single([single_emb1, single_emb2], single_img, single_path)
                else:
                    img, path = t
                    # send batch to device
                    img = img.cuda()
                    # forward pass
                    emb = self.model(img)
                    # iterate over the batch
                    for single_img, single_emb, single_path in zip(img, emb, path):
                        self.process_single([single_emb], single_img, single_path)

    def process_single(self, emb_arr, img, path):
        # predictions to save to h5 file
        out_file = os.path.splitext(path)[0] + '_predictions.h5'
        out_file = os.path.join(self.output_dir, os.path.split(out_file)[1])
        for i, se in enumerate(emb_arr):
            # save PNG with PCA projected embeddings
            embeddings_numpy = np.squeeze(se.cpu().numpy())
            rgb_img = pca_project(embeddings_numpy)
            Image.fromarray(np.rollaxis(rgb_img, 0, 3)).save(os.path.splitext(out_file)[0] + f'_{i + 1}.png')

        with h5py.File(out_file, 'w') as f:
            print(f'Saving output to {out_file}')
            f.create_dataset('raw', data=img.cpu().numpy(), compression='gzip')
            for i, emb in enumerate(emb_arr):
                f.create_dataset(f'embeddings{i + 1}', data=emb.cpu().numpy(), compression='gzip')
