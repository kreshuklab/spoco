import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from spoco.model import WGANDiscriminator
from spoco.utils import EmbeddingsTensorboardFormatter, RunningAverage, save_checkpoint, create_optimizer
from spoco.wgantrainer import WGANTrainer


class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, train_loader, val_loader, checkpoint_dir,
                 max_num_iterations, validate_after_iters, log_after_iters,
                 num_iterations=1, num_epoch=0, eval_score_higher_is_better=True,
                 best_eval_score=None, tensorboard_formatter=None):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    def fit(self):
        while not self.train():
            # train for one epoch
            print('Epoch: ', self.num_epoch)
            self.num_epoch += 1

        print('Stopping criterion is satisfied!')

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()

        # sets the model in training mode
        self.model.train()

        for im_q, im_k, target in self.train_loader:
            print(f'Training iteration: [{self.num_iterations}/{self.max_num_iterations}]. Epoch: {self.num_epoch}')

            im_q = im_q.to(self.device)
            im_k = im_k.to(self.device)
            target = target.to(self.device)

            # forward pass through SpocoUNet
            emb_q, emb_k = self.model(im_q, im_k)
            emb_k = emb_k.detach()
            input = (im_q, im_k)
            output = (emb_q, emb_k)

            # compute embedding consistency loss
            loss = self.loss_criterion((emb_q, emb_k), target)

            b_size = self._batch_size(im_q)
            train_losses.update(loss.item(), b_size)

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score = self.validate()
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                self.scheduler.step(eval_score)

                # log current learning rate in tensorboard
                self._log_lr()

                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                eval_score = self.eval_criterion(emb_q, target)
                train_eval_scores.update(eval_score.item(), b_size)

                # log stats, params and images
                print(f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_images(input, target, output, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            print(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            print(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        print('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()

        with torch.no_grad():
            for i, (im_q, im_k, target) in enumerate(self.val_loader):
                print(f'Validation iteration {i}')

                im_q = im_q.to(self.device)
                im_k = im_k.to(self.device)
                target = target.to(self.device)

                # forward pass through SpocoUNet
                emb_q, emb_k = self.model(im_q, im_k)

                # compute embedding consistency loss
                loss = self.loss_criterion((emb_q, emb_k), target)

                b_size = self._batch_size(im_q)
                val_losses.update(loss.item(), b_size)

                should_log = True
                # TODO:
                # if isinstance(self.val_loader.dataset, HDF5Dataset):
                #    should_log = random.random() < 0.1

                if should_log:
                    self._log_images(im_q, target, (emb_q, emb_k), 'val_')

                # use only emb_q for validation
                eval_score = self.eval_criterion(emb_q, target)
                val_scores.update(eval_score.item(), b_size)

            self._log_stats('val', val_losses.avg, val_scores.avg)
            print(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            print(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        print('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)


def create_trainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device, train_loader, val_loader,
                   args):
    is3d = args.model_name == 'UNet3D'
    tensorboard_formatter = EmbeddingsTensorboardFormatter(plot_variance=True, is3d=is3d)

    if not args.gan:
        print('Standard SPOCO training')
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=loss_criterion,
            eval_criterion=eval_criterion,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=args.checkpoint_dir,
            max_num_iterations=args.max_num_iterations,
            validate_after_iters=args.validate_after_iters,
            log_after_iters=args.log_after_iters,
            tensorboard_formatter=tensorboard_formatter
        )
    else:
        print('Training SPOCO in adversarial mode')
        if args.ds_name == 'cvppp':
            patch_shape = (448, 448)
        elif args.ds_name == 'dsb':
            patch_shape = (256, 256)

        critic = WGANDiscriminator(
            # critic takes a single channel input, i.e. the instance pmaps given by the differentiable instance extraction
            in_channels=1,
            f_maps=args.model_feature_maps,
            # don't use normalization layers in the critic
            layer_order='cl',
            is3d=is3d,
            patch_shape=patch_shape
        )

        critic = critic.to(device)

        D_optimizer = create_optimizer(0.0001, critic, betas=(0.5, 0.9))
        trainer = WGANTrainer(
            G=model,
            D=critic,
            G_optimizer=optimizer,
            D_optimizer=D_optimizer,
            G_lr_scheduler=lr_scheduler,
            G_loss_criterion=loss_criterion,
            G_eval_criterion=eval_criterion,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=args.checkpoint_dir,
            max_num_iterations=args.max_num_iterations,
            gp_lambda=args.gradient_penalty_weight,
            gan_loss_weight=args.gan_loss_weight,
            critic_iters=args.critic_iters,
            kernel_threshold=args.kernel_threshold,
            validate_after_iters=args.validate_after_iters,
            log_after_iters=args.log_after_iters,
            bootstrap_G=args.bootstrap_embeddings,
            tensorboard_formatter=tensorboard_formatter
        )

    return trainer
