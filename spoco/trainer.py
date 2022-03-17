import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from skimage.color import label2rgb
from torch.utils.tensorboard import SummaryWriter

from spoco.datasets.utils import create_train_val_loaders
from spoco.losses import create_loss
from spoco.model import create_model, get_number_of_learnable_parameters, UNet3D
from spoco.utils import RunningAverage, save_checkpoint, create_optimizer, pca_project, minmax_norm


class AbstractTrainer:
    def __init__(self, args):
        # create model
        model = create_model(args)
        model.cuda(args.gpu)
        print(model)
        print(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
        self.is3d = isinstance(model, UNet3D)
        self.model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        print(f"Using {torch.cuda.device_count()} GPUs for training")

        self.loss = create_loss(args.loss_delta_var, args.loss_delta_dist, args.loss_alpha, args.loss_beta,
                                args.loss_gamma, args.loss_unlabeled_push, args.loss_instance_weight,
                                args.loss_consistency_weight, args.kernel_threshold, args.instance_loss, args.spoco)
        self.loss.cuda(args.gpu)
        print(f"Loss function: {self.loss}")
        # create optimizer
        self.optimizer = create_optimizer(model, args.learning_rate, args.weight_decay, args.betas)
        self.lr = args.learning_rate
        self.cos = args.cos
        self.schedule = args.schedule
        if not self.cos:
            assert args.schedule is not None and len(args.schedule) > 0 and all(0 < m < 1 for m in args.schedule)
        # create dataloaders
        self.train_loader, self.val_loader = create_train_val_loaders(args)

        self.checkpoint_dir = args.checkpoint_dir
        self.save_all_checkpoints = args.save_all_checkpoints
        self.max_num_iterations = args.max_num_iterations
        self.max_num_epochs = args.max_num_epochs

        assert self.max_num_iterations is not None or self.max_num_epochs is not None

        if self.max_num_epochs is None:
            self.max_num_epochs = torch.cuda.device_count() * args.batch_size * self.max_num_iterations // len(
                self.train_loader.dataset) + 1
            print('Computed max number of epochs:', self.max_num_epochs)

        if self.max_num_iterations is None:
            self.max_num_iterations = self.max_num_epochs * len(self.train_loader.dataset) // (
                    args.batch_size * torch.cuda.device_count())
            print('Computed max number of iterations:', self.max_num_iterations)

        self.log_after_iters = args.log_after_iters
        self.num_iterations = 0
        self.best_validation_loss = torch.finfo().max
        self.gpu = args.gpu
        self.rank = args.rank
        self.writer = None
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, 'logs'))

    def adjust_learning_rate(self, epoch):
        lr = self.lr
        if self.cos:
            # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.max_num_epochs))
        else:
            # multistep lr schedule
            for milestone in self.schedule:
                lr *= 0.1 if epoch >= int(milestone * self.max_num_epochs) else 1.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        for epoch in range(self.max_num_epochs):
            print(f'Epoch [{epoch} / {self.max_num_epochs}]')
            self.train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)
            self.adjust_learning_rate(epoch)
            if self.rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', lr, self.num_iterations)

            # train for one epoch
            should_stop = self.train_epoch()

            # evaluate on the validation set
            self.model.eval()
            validation_loss = self.validate()
            self.model.train()
            # save checkpoint
            is_best = validation_loss < self.best_validation_loss
            if is_best:
                self.best_validation_loss = validation_loss

            if self.rank == 0:
                if self.save_all_checkpoints:
                    checkpoint_name = 'checkpoint_{:05d}.pytorch'.format(epoch)
                else:
                    checkpoint_name = 'last_checkpoint.pytorch'
                checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name)
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'num_iterations': self.num_iterations,
                        'model_state_dict': self.model.state_dict(),
                        'best_validation_loss': self.best_validation_loss,
                        'optimizer': self.optimizer.state_dict()
                    },
                    is_best=is_best,
                    filename=checkpoint_file
                )
                print(f'Checkpoint saved: {checkpoint_file}. Best: {is_best}')

            if should_stop:
                print('Training finished!')
                return

        print('Training finished!')

    def train_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def log_images(self, input, target, output, prefix, log_probab=1.0):

        if random.random() < log_probab:
            # iterate over the batch
            for i, (input_img, target_img, output_img) in enumerate(zip(input, target, output)):
                if self.is3d:
                    # show mid-slice
                    z = input_img.size(1) // 2
                    input_img = input_img[:, z, ...]
                    output_img = output_img[:, z, ...]
                    target_img = target_img[:, z, ...]

                # PCA project the output
                input_img = minmax_norm(input_img.detach().cpu().numpy())
                output_img = pca_project(output_img.detach().cpu().numpy())
                target_img = target_img.detach().cpu().numpy().astype(np.uint8)
                target_img = label2rgb(target_img, bg_label=0)
                self.writer.add_image(prefix + f'{i}_input', input_img, self.num_iterations)
                self.writer.add_image(prefix + f'{i}_output', output_img, self.num_iterations)
                self.writer.add_image(prefix + f'{i}_target', target_img, self.num_iterations, dataformats='HWC')


class SpocoTrainer(AbstractTrainer):
    def train_epoch(self):
        train_losses = RunningAverage()

        for im_f, im_g, target in self.train_loader:
            if self.num_iterations >= self.max_num_iterations:
                print('Reached max number of iterations.')
                return True

            print(f'Training iteration {self.num_iterations}')
            im_f = im_f.cuda(self.gpu, non_blocking=True)
            im_g = im_g.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)

            # forward pass through SpocoUNet
            emb_f, emb_g = self.model(im_f, im_g)
            emb_g = emb_g.detach()

            # compute the loss
            loss = self.loss((emb_f, emb_g), target)
            train_losses.update(loss.item(), im_f.size(0))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.log_after_iters == 0:
                print(f'Training loss: {train_losses.avg}')
                if self.rank == 0:
                    self.writer.add_scalar('train_loss_avg', train_losses.avg, self.num_iterations)
                    self.log_images(im_f, target, emb_f, prefix='train_')

            self.num_iterations += 1

        return False

    def validate(self):
        with torch.no_grad():
            val_losses = RunningAverage()
            dataset_size = len(self.val_loader.dataset)
            # log ~10 batches per epoch
            log_probab = 10 * self.val_loader.batch_size / dataset_size
            with torch.no_grad():
                for i, (im_f, im_g, target) in enumerate(self.val_loader):
                    print(f'Validation iteration {i}')
                    im_f = im_f.cuda(self.gpu, non_blocking=True)
                    im_g = im_g.cuda(self.gpu, non_blocking=True)
                    target = target.cuda(self.gpu, non_blocking=True)

                    # forward pass through SpocoUNet
                    emb_f, emb_g = self.model(im_f, im_g)
                    # compute the loss
                    loss = self.loss((emb_f, emb_g), target)
                    val_losses.update(loss.item(), im_f.size(0))

                    if self.rank == 0:
                        self.log_images(im_f, target, emb_f, prefix=f'val_{i}_', log_probab=log_probab)

                if self.rank == 0:
                    self.writer.add_scalar('val_loss_avg', val_losses.avg, self.num_iterations)

                print(f'Validation finished. Loss: {val_losses.avg}')
                return val_losses.avg


class UNetTrainer(AbstractTrainer):
    def train_epoch(self):
        train_losses = RunningAverage()

        for input, target in self.train_loader:
            if self.num_iterations >= self.max_num_iterations:
                print('Reached max number of iterations.')
                return True

            print(f'Training iteration {self.num_iterations}')
            input = input.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)
            # forward pass
            output = self.model(input)
            # compute the loss
            loss = self.loss(output, target)
            train_losses.update(loss.item(), input.size(0))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.log_after_iters == 0:
                print(f'Training loss: {train_losses.avg}')
                if self.rank == 0:
                    self.writer.add_scalar('train_loss_avg', train_losses.avg, self.num_iterations)
                    self.log_images(input, target, output, prefix='train_')

            self.num_iterations += 1

        return False

    def validate(self):
        val_losses = RunningAverage()
        dataset_size = len(self.val_loader.dataset)
        # log ~10 batches per epoch
        log_probab = 10 * self.val_loader.batch_size / dataset_size
        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                print(f'Validation iteration {i}')
                input = input.cuda(self.gpu, non_blocking=True)
                target = target.cuda(self.gpu, non_blocking=True)

                # forward pass
                output = self.model(input)
                # compute the loss
                loss = self.loss(output, target)
                val_losses.update(loss.item(), input.size(0))

                if self.rank == 0:
                    self.log_images(input, target, output, prefix=f'val_{i}_', log_probab=log_probab)

            if self.rank == 0:
                self.writer.add_scalar('val_loss_avg', val_losses.avg, self.num_iterations)

            print(f'Validation finished. Loss: {val_losses.avg}')
            return val_losses.avg
