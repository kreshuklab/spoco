import argparse
import builtins
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spoco.trainer import SpocoTrainer, UNetTrainer
from spoco.utils import SUPPORTED_DATASETS

parser = argparse.ArgumentParser(description='SPOCO train')
parser.add_argument('--manual-seed', type=int, default=None, help="RNG seed for deterministic training")

# dataset config
parser.add_argument('--ds-name', type=str, default='cvppp', choices=SUPPORTED_DATASETS,
                    help=f'Name of the dataset from: {SUPPORTED_DATASETS}')
parser.add_argument('--ds-path', type=str, required=True, help='Path to the dataset root directory')
parser.add_argument('--things-class', type=str, help='Cityscapes instance class. If None, train with all things classes',
                    default=None)
parser.add_argument('--instance-ratio', type=float, default=None,
                    help='ratio of ground truth instances that should be taken for training')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=4)

# model config
parser.add_argument('--model-name', type=str, default="UNet2D", help="UNet2D or UNet3D")
parser.add_argument('--model-in-channels', type=int, default=3)
parser.add_argument('--model-out-channels', type=int, default=16, help="Embedding space dimension")
parser.add_argument('--model-feature-maps', type=int, nargs="+", default=[16, 32, 64, 128, 256, 512],
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
parser.add_argument('--instance-loss', type=str, default='dice',
                    help="Type of the instance-based loss (dice/affinity/bce")
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
parser.add_argument('--schedule', type=float, nargs="+", help="Multistep LR schedule")
parser.add_argument('--cos', action='store_true', default=False, help="Use cosine learning rate scheduler")

# trainer config
parser.add_argument('--spoco', action='store_true', default=False, help="Indicate SPOCO training with consistency loss")
parser.add_argument('--save-all-checkpoints', action='store_true', default=False, help="Save checkpoint after every epoch")
parser.add_argument('--checkpoint-dir', type=str, required=True, help="Model and tensorboard logs directory")
parser.add_argument('--log-after-iters', type=int, required=True,
                    help="Number of iterations between tensorboard logging")
parser.add_argument('--max-num-iterations', type=int, default=None, help="Maximum number of iterations")
parser.add_argument('--max-num-epochs', type=int, default=None, help="Maximum number of epochs")

# distributed settings
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--node-rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--master-addr', default='localhost', type=str, help='IP addr of the master node')
parser.add_argument('--master-port', default='12357', type=str, help='port on the master node')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--rank', default=0, type=int)


def setup(rank, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=torch.cuda.device_count() * args.nodes)


def train(gpu, args):
    rank = args.node_rank * torch.cuda.device_count() + gpu
    print(f'Running DDP training on rank {rank}. GPU id {gpu}.')
    args.rank = rank
    args.gpu = gpu
    # setup the process group
    setup(rank, args)
    torch.cuda.set_device(gpu)
    # disable logging for non-master node
    if gpu != 0:
        def fake_print(*args):
            pass

        builtins.print = fake_print

    # create trainer
    if args.spoco:
        trainer = SpocoTrainer(args)
    else:
        trainer = UNetTrainer(args)
    print(f'Starting training')
    trainer.train()


def main():
    args = parser.parse_args()
    print('ARGS:', args)

    if not torch.cuda.is_available():
        raise RuntimeError('Only GPU training is supported')

    seed = args.manual_seed
    if seed is not None:
        print(f'Seed the RNG for all devices with {seed}')
        random.seed(seed)
        torch.manual_seed(seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        print('Using CuDNN deterministic setting. This may slow down the training!')

    nprocs = torch.cuda.device_count()
    mp.spawn(train, args=(args,), nprocs=nprocs)


if __name__ == '__main__':
    main()
