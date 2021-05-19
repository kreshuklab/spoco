import argparse
import random

import torch

from spoco.datasets.utils import create_train_val_loaders
from spoco.losses import create_loss
from spoco.metrics import create_eval_metric
from spoco.model import get_number_of_learnable_parameters, create_model
from spoco.trainer import create_trainer
from spoco.utils import create_optimizer, create_lr_scheduler, SUPPORTED_DATASETS

parser = argparse.ArgumentParser(description='SPOCO train')
parser.add_argument('--manual-seed', type=int, default=None, help="RNG seed for deterministic training")

# dataset config
parser.add_argument('--ds-name', type=str, default='cvppp', choices=SUPPORTED_DATASETS,
                    help=f'Name of the dataset from: {SUPPORTED_DATASETS}')
parser.add_argument('--ds-path', type=str, required=True, help='Path to the dataset root directory')
parser.add_argument('--instance-ratio', type=float, default=None,
                    help='ratio of ground truth instances that should be taken for training')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=8)

# model config
parser.add_argument('--model-name', type=str, default="UNet2D", help="UNet2D or UNet3D")
parser.add_argument('--model-in-channels', type=int, default=3)
parser.add_argument('--model-out-channels', type=int, default=16, help="Embedding space dimension")
parser.add_argument('--model-feature-maps', type=int, nargs="+", default=[32, 64, 128, 256, 512],
                    help="Number of features at each level on the encoder path")
parser.add_argument('--model-layer-order', type=str, default="bcr",
                    help="Determines the order of operations for SingleConv layer; 'bcr' means Batchnorm+Conv+ReLU")
parser.add_argument('--momentum', type=float, default=0.999)

# loss definition
parser.add_argument('--loss-delta-var', type=float, default=0.5, help="Pull force hinge")
parser.add_argument('--loss-delta-dist', type=float, default=2.0, help="Push force hinge")
parser.add_argument('--loss-alpha', type=float, default=1.0, help="Pull force term weight")
parser.add_argument('--loss-beta', type=float, default=1.0, help="Push force term weight")
parser.add_argument('--loss-gamma', type=float, default=0.001, help="Regularizer term weight")
parser.add_argument('--loss-unlabeled-push', type=float, default=0.0,
                    help="Unlabeled region push force weight. If greater than 0 then sparse object training mode"
                         "is assumed and 0-label corresponds to the unlabeled region, i.e. no pull force applied to 0-label")
parser.add_argument('--loss-instance-weight', type=float, default=1.0, help="Instance-based term weight")
parser.add_argument('--loss-consistency-weight', type=float, default=1.0, help="Embeddings consistency term weight")
parser.add_argument('--kernel-threshold', type=float, default=0.5,
                    help="Kernel value for points which are delta-var away from the anchor embedding")

# optimizer
parser.add_argument('--learning-rate', type=float, default=0.0002, help="Initial learning rate")
parser.add_argument('--weight-decay', type=float, default=0.00001, help="Weight decay regularization")
parser.add_argument('--betas', type=float, nargs="+", default=[0.9, 0.999], help="Adam optimizer params")
parser.add_argument('--patience', type=int, default=10, help="Learning rate scheduler patience")
parser.add_argument('--lr-factor', type=float, default=0.2, help="Learning rate scheduler factor")

# trainer config
parser.add_argument('--checkpoint-dir', type=str, required=True, help="Model and tensorboard logs directory")
parser.add_argument('--log-after-iters', type=int, required=True,
                    help="Number of iterations between tensorboard logging")
parser.add_argument('--validate-after-iters', type=int, required=True,
                    help="Number of iterations between validation runs")
parser.add_argument('--max-num-iterations', type=int, required=True, help="Maximum number of iterations")

# WGAN training
parser.add_argument('--gan', action='store_true', help='Train in GAN setting')
parser.add_argument('--bootstrap-embeddings', type=int, default=None,
                    help='Number of iters to bootstrap embedding model (Generator)')
parser.add_argument('--gan_loss_weight', type=float, default=0.1, help='Weighting applied to the WGAN loss term')
parser.add_argument('--critic-iters', type=int, default=2, help='Number of critic iters per generator iters')


def main():
    args = parser.parse_args()

    manual_seed = args.manual_seed
    if manual_seed is not None:
        print(f'Seed the RNG for all devices with {manual_seed}')
        random.seed(args.seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # create model
    model = create_model(args)
    device_str = "cuda" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Sending the model to '{device}'")
    model = model.to(device)
    print(model)
    print(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # initialize loss
    loss_criterion = create_loss(args.loss_delta_var, args.loss_delta_dist,
                                 args.loss_alpha, args.loss_beta, args.loss_gamma,
                                 args.loss_unlabeled_push, args.loss_instance_weight,
                                 args.loss_consistency_weight, args.kernel_threshold)
    loss_criterion = loss_criterion.to(device)
    print(f'Using loss function: {loss_criterion}')

    # init eval criterion
    eval_criterion = create_eval_metric(args.ds_name, args.loss_delta_var)
    print(f'Using eval criterion: {eval_criterion}')

    # create optimizer
    optimizer = create_optimizer(args.learning_rate, model, args.weight_decay, args.betas)
    # we use DiceScore and AveragePrecision so higher is always better, i.e. mode='max'
    lr_scheduler = create_lr_scheduler(optimizer, args.patience, mode='max', factor=args.lr_factor)

    # create dataloaders
    print(f'Loading dataset from: {args.ds_path}')
    train_loader, val_loader = create_train_val_loaders(args.ds_name, args.ds_path, args.batch_size, args.num_workers,
                                                        args.instance_ratio, manual_seed)

    # create trainer
    trainer = create_trainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device, train_loader,
                             val_loader, args)
    # start training
    trainer.fit()


if __name__ == '__main__':
    main()
