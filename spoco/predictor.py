import os
from concurrent import futures

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from spoco.utils import pca_project


def save_batch(output_dir, emb, img, path):
    for single_img, single_emb, single_path in zip(img, emb, path):
        save_predictions(output_dir, single_emb, single_img, single_path)


def save_predictions(output_dir, emb, img, path):
    # predictions to save to h5 file
    out_file = os.path.splitext(path)[0] + '_predictions.h5'
    pred_filename = os.path.basename(out_file)
    out_file = os.path.join(output_dir, pred_filename)

    with h5py.File(out_file, 'w') as f:
        # print(f'Saving output to {out_file}')
        f.create_dataset('raw', data=img, compression='gzip')
        f.create_dataset(f'embeddings', data=emb, compression='gzip')

        # save PNG with PCA projected embeddings
        emb_np = np.squeeze(emb)
        rgb_img = pca_project(emb_np)
        Image.fromarray(np.rollaxis(rgb_img, 0, 3)).save(os.path.splitext(out_file)[0] + '.png')


class EmbeddingsPredictor:
    def __init__(self, model, test_loader, output_dir, spoco):
        self.model = model
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.spoco = spoco

    def predict(self):
        # set the model in evaluation mode explicitly
        self.model.eval()

        # initial process pool for saving results to disk
        executor = futures.ProcessPoolExecutor(max_workers=32)

        # run predictions on the entire test_set
        with torch.no_grad():
            for t in tqdm(self.test_loader):
                if self.spoco:
                    img, img2, path = t
                    # send batch to device
                    img, img2 = img.cuda(), img2.cuda()
                    # forward pass
                    emb, _ = self.model(img, img2)
                else:
                    img, path = t
                    # send batch to device
                    img = img.cuda()
                    # forward pass
                    emb = self.model(img)

                # save predictions to disk
                executor.submit(
                    save_batch,
                    self.output_dir,
                    emb.cpu().numpy(),
                    img.cpu().numpy(),
                    path
                )

        print('Waiting for all predictions to be saved to disk...')
        executor.shutdown(wait=True)
