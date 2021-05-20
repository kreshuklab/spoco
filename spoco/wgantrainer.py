import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import autograd

from spoco.losses import expand_as_one_hot, compute_cluster_means
from spoco.utils import RunningAverage, save_checkpoint, GaussianKernel


class WGANTrainer:
    def __init__(self, G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion, G_eval_criterion, device,
                 train_loader, val_loader, checkpoint_dir, max_num_iterations, gp_lambda, gan_loss_weight, critic_iters,
                 kernel_threshold, validate_after_iters, log_after_iters, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None, bootstrap_G=None, tensorboard_formatter=None):
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.checkpoint_dir = checkpoint_dir
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.device = device
        self.G_eval_criterion = G_eval_criterion
        self.G_loss_criterion = G_loss_criterion
        self.G_lr_scheduler = G_lr_scheduler

        self.gan_loss_weight = gan_loss_weight
        self.gp_lambda = gp_lambda
        self.critic_iters = critic_iters
        self.bootstrap_G = bootstrap_G

        self.best_eval_score = best_eval_score
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.num_epoch = num_epoch
        self.max_num_iterations = max_num_iterations
        self.num_iterations = num_iterations
        self.G_iterations = 1
        self.D_iterations = 1
        self.log_after_iters = log_after_iters
        self.validate_after_iters = validate_after_iters
        self.kernel_threshold = kernel_threshold
        self.tensorboard_formatter = tensorboard_formatter

        # create mask extractor
        dist_to_mask = GaussianKernel(delta_var=G_loss_criterion.delta_var, pmaps_threshold=kernel_threshold)
        self.fake_mask_extractor = TargetMeanMaskExtractor(dist_to_mask)

        print('CRITIC/DISCRIMINATOR')
        print(self.D)
        print(f'gan_loss_weight: {gan_loss_weight}')
        print(f'bootstrap_G: {bootstrap_G}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

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
        # keeps running average of the contrastive loss
        emb_losses = RunningAverage()
        # keeps track of the generator part of the GAN loss
        G_losses = RunningAverage()
        # keeps track of the discriminator part of the GAN loss
        D_losses = RunningAverage()
        # keeps track of the eval score of the generator (i.e. embedding network)
        G_eval_scores = RunningAverage()
        # keeps track of the estimate of Wasserstein Distance
        Wasserstein_dist = RunningAverage()

        # sets the model in training mode
        self.G.train()
        self.D.train()

        for im_q, im_k, target in self.train_loader:
            print(f'Training iteration: [{self.num_iterations}/{self.max_num_iterations}]. Epoch: {self.num_epoch}')

            im_q = im_q.to(self.device)
            im_k = im_k.to(self.device)
            target = target.to(self.device)

            if self.num_iterations % self._D_iters() == 0:
                print('Train G')

                self._freeze_D()
                self.G_optimizer.zero_grad()

                # forward pass through embedding network (generator)
                emb_q, emb_k = self.G(im_q, im_k)

                # compute embedding loss
                emb_loss = self.G_loss_criterion((emb_q, emb_k), target)
                emb_losses.update(emb_loss.item(), self.batch_size(emb_q))

                if self.bootstrap_G is not None and self.num_iterations <= self.bootstrap_G:
                    # if we're in the bootstrap phase, optimize only emb loss
                    emb_loss.backward()
                    self.G_optimizer.step()
                    self.num_iterations += 1
                    continue

                # compute GAN loss
                # real_masks are not used in the G update phase, but are needed for tensorboard logging later
                real_masks = create_real_masks(target)
                fake_masks = self.fake_mask_extractor(emb_q, target)

                if fake_masks is None:
                    # skip background patches and backprop only through embedding loss
                    emb_loss.backward()
                    self.G_optimizer.step()
                    self.num_iterations += 1
                    continue

                # train using fake masks; make sure to minimize -G_loss
                G_loss = -self.D(fake_masks).mean(dim=0)
                G_losses.update(G_loss.item(), self.batch_size(fake_masks))

                # compute combined embedding and GAN loss;
                combined_loss = emb_loss + self.gan_loss_weight * G_loss
                combined_loss.backward()

                self.G_optimizer.step()

                self._unfreeze_D()

                self.G_iterations += 1
            else:
                print('Train D')
                self.D_optimizer.zero_grad()

                with torch.no_grad():
                    # forward pass through embedding network (generator)
                    # make sure the G is frozen
                    emb_q, emb_k = self.G(im_q, im_k)

                emb_q = emb_q.detach()  # make sure that G is not updated

                # create real and fake instance masks
                real_masks = create_real_masks(target)
                fake_masks = self.fake_mask_extractor(emb_q, target)

                if real_masks is None or fake_masks is None:
                    self.num_iterations += 1
                    # skip background patches
                    continue

                if real_masks.size()[0] >= 40:
                    self.num_iterations += 1
                    # skip if there are too many instances in the patch in order to prevent CUDA OOM errors
                    continue

                # get the critic value for real masks
                D_real = self.D(real_masks).mean(dim=0)

                # get the critic value for fake masks
                D_fake = self.D(fake_masks).mean(dim=0)

                # train with gradient penalty
                gradient_penalty = self._calc_gp(real_masks, fake_masks)

                # we want to maximize the WGAN value function D(real) - D(fake), i.e.
                # we want to minimize D(fake) - D(real)
                D_loss = D_fake - D_real + self.gp_lambda * gradient_penalty
                # backprop
                D_loss.backward()

                # update D's weights
                self.D_optimizer.step()

                n_batch = 2 * self.batch_size(fake_masks)
                D_losses.update(D_loss.item(), n_batch)

                Wasserstein_D = D_real - D_fake
                Wasserstein_dist.update(Wasserstein_D.item(), n_batch)

                self.D_iterations += 1

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.G.eval()
                # evaluate on validation set
                eval_score = self.validate()
                # set the model back to training mode
                self.G.train()

                # adjust learning rate if necessary
                if self.G_lr_scheduler is not None:
                    self.G_lr_scheduler.step(eval_score)
                # log current learning rate in tensorboard
                self._log_G_lr()
                # remember best validation metric
                is_best = self.is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                eval_score = self.G_eval_criterion(emb_q, target)
                G_eval_scores.update(eval_score.item(), self.batch_size(im_q))

                # log stats, params and images
                print(
                    f'Training stats. Embedding Loss: {emb_losses.avg}. GAN Loss: {G_losses.avg}. '
                    f'Discriminator Loss: {D_losses.avg}. Evaluation score: {G_eval_scores.avg}')

                self.writer.add_scalar('train_embedding_loss', emb_losses.avg, self.num_iterations)
                self.writer.add_scalar('train_GAN_loss', G_losses.avg, self.num_iterations)
                self.writer.add_scalar('train_D_loss', D_losses.avg, self.num_iterations)
                self.writer.add_scalar('Wasserstein_distance', Wasserstein_dist.avg, self.num_iterations)

                inputs_map = {
                    'inputs': (im_q, im_k),
                    'targets': target,
                    'predictions': (emb_q, emb_k)
                }
                self._log_images(inputs_map)
                # log masks if we're not during G training phase
                if self.num_iterations % (self.critic_iters + 1) != 0:
                    inputs_map = {
                        'real_masks': real_masks,
                        'fake_masks': fake_masks
                    }
                    self._log_images(inputs_map)

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def _D_iters(self):
        if self.bootstrap_G is not None:
            # train generator for `bootstrap_G` iterations first
            if self.num_iterations <= self.bootstrap_G:
                # just return `num_iterations` so that we always land in the G training phase
                return self.num_iterations
            else:
                return self.critic_iters + 1
        else:
            return self.critic_iters + 1

    def _calc_gp(self, real_masks, fake_masks):
        # align real and fake masks
        n_batch = min(real_masks.size(0), fake_masks.size(0))

        real_masks = real_masks[:n_batch]
        fake_masks = fake_masks[:n_batch]

        a_shape = [1] * real_masks.dim()
        a_shape[0] = n_batch

        alpha = torch.rand(a_shape)
        alpha = alpha.expand_as(real_masks)
        alpha = alpha.to(real_masks.device)

        interpolates = alpha * real_masks + ((1 - alpha) * fake_masks)
        interpolates.requires_grad = True

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(real_masks.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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

                emb_q, emb_k = self.G(im_q, im_k)
                loss = self.G_loss_criterion((emb_q, emb_k), target)
                val_losses.update(loss.item(), self.batch_size(im_q))
                eval_score = self.G_eval_criterion(emb_q, target)
                val_scores.update(eval_score.item(), self.batch_size(im_q))

            self.writer.add_scalar('val_embedding_loss', val_losses.avg, self.num_iterations)
            self.writer.add_scalar('val_eval_score_avg', val_scores.avg, self.num_iterations)
            print(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            print(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            print(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.G_optimizer.param_groups[0]['lr']
        if lr < min_lr:
            print(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    @staticmethod
    def batch_size(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x[0].size(0)
        else:
            return x.size(0)

    def _freeze_D(self):
        # freeze all layers of D
        for p in self.D.parameters():
            p.requires_grad = False

    def _unfreeze_D(self):
        for p in self.D.parameters():
            p.requires_grad = True

    def _log_G_lr(self):
        lr = self.G_optimizer.param_groups[0]['lr']
        self.writer.add_scalar('G_learning_rate', lr, self.num_iterations)

    def _log_images(self, inputs_map):
        assert isinstance(inputs_map, dict)
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='CHW')

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.G, nn.DataParallel):
            G_state_dict = self.G.module.state_dict()
            D_state_dict = self.D.module.state_dict()
        else:
            G_state_dict = self.G.state_dict()
            D_state_dict = self.D.state_dict()

        # save generator and discriminator state + metadata
        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': G_state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.G_optimizer.state_dict(),
            'device': str(self.device),
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            # discriminator
            'D_model_state_dict': D_state_dict,
            'D_optimizer_state_dict': self.D_optimizer.state_dict()
        },
            is_best=is_best,
            checkpoint_dir=self.checkpoint_dir)


class AbstractMaskExtractor:
    def __init__(self, dist_to_mask):
        """
        Base class for extracting the 'fake' masks given the embeddings.

        Args:
            dist_to_mask (Callable): function which converts the distance map to an instance map
        """
        self.dist_to_mask = dist_to_mask

    def __call__(self, embeddings, labels=None):
        """
        Computes the instance map given the embeddings tensor (no batch dim) and the optional labels (no batch dim)

        Args:

            embeddings: NxExSPATIAL embedding tensor
            labels: (optional) tensor containing the instance ground truth

        Returns:
            list of instance masks
        """

        fake_masks = []
        # iterate over the batch
        for i, emb in enumerate(embeddings):
            # extract the masks from a single batch instance
            fms = self.extract_masks(emb, labels[i] if labels is not None else None)
            fake_masks.extend(fms)

        if len(fake_masks) == 0:
            return None

        fake_masks = torch.stack(fake_masks).to(embeddings.device)
        return fake_masks

    def extract_masks(self, embeddings, labels=None):
        """Extract mask from a single batch instance"""
        raise NotImplementedError


class TargetBasedMaskExtractor(AbstractMaskExtractor):
    """
    Extracts the instance masks given the embeddings and the anchor_embeddings_extractor, which extracts the
    anchor embeddings given the target labeling.
    """

    def __init__(self, dist_to_mask, anchor_embeddings_extractor):
        super().__init__(dist_to_mask)
        self.anchor_embeddings_extractor = anchor_embeddings_extractor

    def extract_masks(self, emb, tar=None):
        assert tar is not None

        anchor_embeddings = self.anchor_embeddings_extractor(emb, tar)

        results = []
        for i, anchor_emb in enumerate(anchor_embeddings):
            if i == 0:
                # ignore 0-label
                continue

            # compute distance map; embeddings is ExSPATIAL, anchor_embeddings is ExSINGLETON_SPATIAL, so we can just broadcast
            dist_to_mean = torch.norm(emb - anchor_emb, 'fro', dim=0)
            # convert distance map to instance pmaps
            inst_pmap = self.dist_to_mask(dist_to_mean)
            # add channel dim and save fake masks
            results.append(inst_pmap.unsqueeze(0))

        return results


class TargetMeanMaskExtractor(TargetBasedMaskExtractor):
    def __init__(self, dist_to_mask):
        super().__init__(dist_to_mask, MeanEmbeddingAnchor())


class TargetRandomMaskExtractor(TargetBasedMaskExtractor):
    def __init__(self, dist_to_mask):
        super().__init__(dist_to_mask, RandomEmbeddingAnchor())


def extract_fake_masks(emb, dist_to_mask, volume_threshold=0.1, max_instances=40, max_iterations=100):
    """
    Extracts instance pmaps given the embeddings. The algorithm works by using a heuristic to find so called
    'anchor embeddings' (think of it as a cluster centers), then for each of the anchors it computes the distance map
    and uses the 'dist_to_mask' kernel in order to convert a given distance map to a given instance pmap.

    Args:
        emb: pixel embeddings (ExSPATIAL), where E is the embedding dim
        dist_to_mask: kernel converting a distance map to instance pmaps
        volume_threshold: percentage of the overall volume that can be left unsegmented
        max_instances: maximum number of instance pmaps to be returned
        max_iterations: maximum number of iterations

    Returns:
        a list of instance pmaps
    """
    # initialize the volume in order to track visited voxels
    visited = torch.ones(emb.shape[1:])

    results = []
    mask_sizes = []
    # check stop criteria
    while visited.sum() > visited.numel() * volume_threshold and len(results) < max_iterations:
        # get voxel coordinates
        z, y, x = torch.nonzero(visited, as_tuple=True)
        ind = torch.randint(len(z), (1,))[0]
        anchor_emb = emb[:, z[ind], y[ind], x[ind]]
        # (E,) -> (E, 1, 1, 1)
        anchor_emb = anchor_emb[..., None, None, None]

        # compute distance map; embeddings is ExSPATIAL, anchor_embeddings is ExSINGLETON_SPATIAL, so we can just broadcast
        dist_to_anchor = torch.norm(emb - anchor_emb, 'fro', dim=0)
        inst_mask = dist_to_anchor < dist_to_mask.delta_var
        # convert distance map to instance pmaps
        inst_pmap = dist_to_mask(dist_to_anchor)

        mask_sizes.append(inst_mask.sum())
        results.append(inst_pmap.unsqueeze(0))

        # update visited array
        visited[inst_mask] = 0

    # get the biggest instances and limit the instances due to OOM errors
    results = [x for _, x in sorted(zip(mask_sizes, results), key=lambda pair: pair[0])]
    results = results[:max_instances]

    return results


def create_real_masks(target):
    real_masks = []
    for tar in target:
        rms = []
        for i in torch.unique(tar):
            if i == 0:
                # ignore 0-label
                continue

            inst_mask = (tar == i).float()
            # add channel dim and save real masks
            rms.append(inst_mask.unsqueeze(0))
            real_masks.extend(rms)

    if len(real_masks) == 0:
        return None

    real_masks = torch.stack(real_masks).to(target.device)
    return real_masks


class MeanEmbeddingAnchor:
    def __call__(self, emb, tar):
        instances = torch.unique(tar)
        C = instances.size(0)

        single_target = expand_as_one_hot(tar.unsqueeze(0), C).squeeze(0)
        single_target = single_target.unsqueeze(1)
        spatial_dims = emb.dim() - 1

        cluster_means, _, _ = compute_cluster_means(emb, single_target, spatial_dims)
        return cluster_means


class RandomEmbeddingAnchor:
    """
    Selects a random pixel inside an instance, gets its embedding and uses is as an anchor embedding
    """

    def __call__(self, emb, tar):
        instances = torch.unique(tar)
        anchor_embeddings = []
        for i in instances:
            indices = torch.nonzero(tar == i, as_tuple=True)
            ind = torch.randint(len(indices[0]), (1,))[0]
            if tar.dim() == 2:
                y, x = indices
                anchor_emb = emb[:, y[ind], x[ind]]
            else:
                z, y, x = indices
                anchor_emb = emb[:, z[ind], y[ind], x[ind]]
            anchor_embeddings.append(anchor_emb)

        result = torch.stack(anchor_embeddings, dim=0).to(emb.device)
        # expand dimensions
        if tar.dim() == 2:
            result = result[..., None, None]
        else:
            result = result[..., None, None, None]
        return result
