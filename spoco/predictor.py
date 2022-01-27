import os

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image

from spoco.transforms import RgbToLabel, Relabel
from spoco.utils import pca_project


class Abstract2DEmbeddingsPredictor:
    def __init__(self, model, test_loader, output_dir, save_gt):
        self.model = model
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.save_gt = save_gt

    def predict(self):
        # set the model in evaluation mode explicitly
        self.model.eval()

        # run predictions on the entire test_set
        with torch.no_grad():
            for img1, img2, path in self.test_loader:
                # send batch to device
                img1.cuda()
                img2.cuda()

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

                        # save ground truth segmentation if needed
                        if self.save_gt:
                            gt = self.load_gt_label(single_path)
                            f.create_dataset('label', data=gt, compression='gzip')

    def load_gt_label(self, img_path):
        raise NotImplementedError


class CVPPPEmbeddingsPredictor(Abstract2DEmbeddingsPredictor):
    def __init__(self, model, test_loader, output_dir, save_gt):
        super().__init__(model, test_loader, output_dir, save_gt)

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


class CityscapesEmbeddingsPredictor(Abstract2DEmbeddingsPredictor):
    def __init__(self, model, test_loader, output_dir, class_name, root_dir, save_gt):
        super().__init__(model, test_loader, output_dir, save_gt)

        self.annotations_base = os.path.join(root_dir, 'gtFine', 'test')
        self.class_name = class_name
        self.label_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(384, 768), interpolation=Image.NEAREST),
                Relabel(run_cc=False),
                torchvision.transforms.ToTensor()
            ]
        )

    def load_gt_label(self, img_path):
        inst_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            self.class_name,
            '1.0',
            os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png"
        )
        mask = Image.open(inst_path)
        return self.label_transform(mask)


def create_predictor(model, test_loader, output_dir, args):
    if args.ds_name == 'cvppp':
        return CVPPPEmbeddingsPredictor(model, test_loader, output_dir, args.save_gt)
    elif args.ds_name == 'cityscapes':
        return CityscapesEmbeddingsPredictor(model, test_loader, output_dir, class_name=args.things_class,
                                             root_dir=args.ds_path, save_gt=args.save_gt)
    else:
        raise RuntimeError(f'Unsupported dataset {args.ds_name}')
