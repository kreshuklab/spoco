import argparse

import os
import torch

from spoco.datasets.utils import create_test_loader
from spoco.predictor import create_predictor
from spoco.utils import SUPPORTED_DATASETS, load_checkpoint
from spoco.model import create_model

parser = argparse.ArgumentParser(description='SPOCO predict')

# dataset config
parser.add_argument('--ds-name', type=str, default='cvppp', choices=SUPPORTED_DATASETS,
                    help=f'Name of the dataset from: {SUPPORTED_DATASETS}')
parser.add_argument('--ds-path', type=str, required=True, help='Path to the dataset root directory')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--output-dir', type=str, default='.', help='Directory where prediction are to be saved')

# model config
parser.add_argument('--model-name', type=str, default="UNet2D", help="UNet2D or UNet3D")
parser.add_argument('--model-path', type=str, required=True, help="Path to the model's checkpoint")
parser.add_argument('--model-in-channels', type=int, default=3)
parser.add_argument('--model-out-channels', type=int, default=16, help="Embedding space dimension")
parser.add_argument('--model-feature-maps', type=int, nargs="+", default=[32, 64, 128, 256, 512],
                    help="Number of features at each level on the encoder path")
parser.add_argument('--model-layer-order', type=str, default="bcr",
                    help="Determines the order of operations for SingleConv layer; 'bcr' means Batchnorm+Conv+ReLU")


def main():
    args = parser.parse_args()

    # load model from checkpoint
    model = create_model(args)
    print(f'Loading model from {args.model_path}...')
    load_checkpoint(args.model_path, model)

    device_str = "cuda" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Sending the model to '{device}'")
    model = model.to(device)

    # create output dir if necessary
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving predictions to: {output_dir}')

    # create test loader
    test_loader = create_test_loader(args)

    # crete predictor
    predictor = create_predictor(model, test_loader, output_dir, device, args)
    # run inference
    predictor.predict()


if __name__ == '__main__':
    main()
