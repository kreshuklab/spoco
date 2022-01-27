import os

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image

from spoco.transforms import RgbToLabel, Relabel
from spoco.utils import pca_project


class Abstract2DEmbeddingsPredictor:
    def __init__(self, model, test_loader, output_dir, device):
        self.model = model
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.device = device

    def predict(self):
        # set the model in evaluation mode explicitly
        self.model.eval()

        # run predictions on the entire test_set
        with torch.no_grad():
            for img1, img2, path in self.test_loader:
                # send batch to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                # forward pass
                emb1, emb2 = self.model(img1, img2)

                for single_img, single_emb1, single_emb2, single_path in zip(img1, emb1, emb2, path):
                    # predictions to save to h5 file
                    out_file = os.path.splitext(single_path)[0] + '_predictions.h5'
                    out_file = os.path.join(self.output_dir, os.path.split(out_file)[1])

                    for i, se in enumerate([single_emb1, single_emb2]):
                        # save PNG with PCA projected embeddings
                        embeddings_numpy = np.squeeze(se.cpu().numpy())
                        rgb_img = pca_project(embeddings_numpy)
                        Image.fromarray(np.rollaxis(rgb_img, 0, 3)).save(os.path.splitext(out_file)[0] + f'_{i+1}.png')

                    with h5py.File(out_file, 'w') as f:
                        print(f'Saving output to {out_file}')
                        f.create_dataset('raw', data=single_img.cpu().numpy(), compression='gzip')
                        f.create_dataset('embeddings1', data=single_emb1.cpu().numpy(), compression='gzip')
                        f.create_dataset('embeddings2', data=single_emb2.cpu().numpy(), compression='gzip')

                        # save ground truth segmentation if provided
                        gt = self.load_gt_label(single_path)
                        if gt is not None:
                            f.create_dataset('label', data=gt, compression='gzip')

    def load_gt_label(self, img_path):
        raise NotImplementedError


class CVPPPEmbeddingsPredictor(Abstract2DEmbeddingsPredictor):
    def __init__(self, model, test_loader, output_dir, device):
        super().__init__(model, test_loader, output_dir, device)

    def load_gt_label(self, img_path):
        base, filename = os.path.split(img_path)
        prefix = filename.split('_')[0]
        label_file = os.path.join(base, prefix + '_label.png')
        if not os.path.exists(label_file):
            # just load foreground mask
            label_file = os.path.join(base, prefix + '_fg.png')
        img = Image.open(label_file).convert('RGB')

        label_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(448, 448), interpolation=Image.NEAREST),
            RgbToLabel(),
            Relabel(run_cc=False)
        ])
        img = label_transform(img)
        return img


def create_predictor(model, test_loader, output_dir, device, args):
    if args.ds_name == 'cvppp':
        return CVPPPEmbeddingsPredictor(model, test_loader, output_dir, device)
