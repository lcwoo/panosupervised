# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os
import random
from abc import ABC
from collections import OrderedDict

import torch

from vidar.utils.config import cfg_has, read_config
from vidar.utils.data import set_random_seed
from vidar.utils.distributed import print0, rank, world_size
from vidar.utils.flip import flip_batch, flip_output
from vidar.utils.logging import pcolor, set_debug
from vidar.utils.networks import load_checkpoint, save_checkpoint, freeze_layers_and_norms
from vidar.utils.setup import setup_arch, setup_datasets, setup_metrics
from vidar.utils.types import is_str


class Wrapper(torch.nn.Module, ABC):
    """
    Trainer class for model optimization and inference

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    ckpt : String
        Name of the model checkpoint to start from
    verbose : Bool
        Print information on screen if enabled
    """
    def __init__(self, cfg, ckpt=None, verbose=False):
        super().__init__()

        if verbose and rank() == 0:
            font = {'color': 'cyan', 'attrs': ('bold', 'dark')}
            print(pcolor('#' * 100, **font))
            print(pcolor('#' * 42 + ' VIDAR WRAPPER ' + '#' * 43, **font))
            print(pcolor('#' * 100, **font))

        # Get configuration
        cfg = read_config(cfg) if is_str(cfg) else cfg
        self.cfg = cfg

        # Data augmentations
        self.flip_lr_prob = cfg_has(cfg.wrapper, 'flip_lr_prob', 0.0)
        self.validate_flipped = cfg_has(cfg.wrapper, 'validate_flipped', False)

        # Set random seed
        set_random_seed(cfg.wrapper.seed + rank())
        set_debug(cfg_has(cfg.wrapper, 'debug', False))

        # Setup architecture, datasets and tasks
        self.arch = setup_arch(cfg.arch, checkpoint=ckpt, verbose=verbose) if cfg_has(cfg, 'arch') else None
        self.datasets, self.datasets_cfg = setup_datasets(
            cfg.datasets, verbose=verbose) if cfg_has(cfg, 'datasets') else (None, None)
        self.metrics = setup_metrics(cfg.evaluation) if cfg_has(cfg, 'evaluation') else {}

        sync_batch_norm = cfg_has(cfg.wrapper, 'sync_batch_norm', False)
        if sync_batch_norm and os.environ['DIST_MODE'] == 'ddp':
            self.arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.arch)

        self.mixed_precision = cfg_has(cfg.wrapper, 'mixed_precision', False)

        self.update_schedulers = None

    def save(self, filename, epoch=None):
        """Save checkpoint"""
        save_checkpoint(filename, self, epoch=epoch)

    def load(self, checkpoint, strict=True, verbose=False):
        """Load checkpoint"""
        load_checkpoint(self, checkpoint, strict=strict, verbose=verbose)

    def train_custom(self, in_optimizers, out_optimizers):
        """Customized training flag for the model"""
        self.arch.train()
        for key in in_optimizers.keys():
            arch = self.arch.module if hasattr(self.arch, 'module') else self.arch
            freeze_layers_and_norms(arch.networks[key], ['ALL'], flag_freeze=False)
        for key in out_optimizers.keys():
            arch = self.arch.module if hasattr(self.arch, 'module') else self.arch
            freeze_layers_and_norms(arch.networks[key], ['ALL'], flag_freeze=True)

    def eval_custom(self):
        """Customized evaluation flag for the model"""
        self.arch.eval()

    def configure_optimizers_and_schedulers(self, verbose=True):
        """Configure depth and pose optimizers and the corresponding scheduler"""

        if not cfg_has(self.cfg, 'optimizers'):
            return None, None

        exclude_params = cfg_has(self.cfg.arch.networks.depth, 'freeze_params', [])
        if verbose:
            font_base = {'color': 'yellow', 'attrs': ('bold', 'dark')}
            font_name = {'color': 'yellow', 'attrs': ('bold',)}

            print0(pcolor('#' * 60, **font_base))
            print0(pcolor('###### Optimizer', **font_base))
            if len(exclude_params):
                print0(pcolor('- Excluded params:', **font_name))
                print0(pcolor('\n'.join(exclude_params), **font_name))
            print0(pcolor('#' * 60, **font_base))

        optimizers = OrderedDict()
        schedulers = OrderedDict()

        for key, val in self.cfg.optimizers.__dict__.items():
            assert key in self.arch.networks, f'There is no network for optimizer {key}'
            params = [p for n, p in self.arch.networks[key].named_parameters()
                            if not any([ep in n for ep in exclude_params])]
            optimizers[key] = {
                'optimizer': getattr(torch.optim, val.name)(**{
                    'lr': val.lr,
                    'weight_decay': cfg_has(val, 'weight_decay', 0.0),
                    'params': params,
                }),
                'settings': {} if not cfg_has(val, 'settings') else val.settings.__dict__
            }
            if cfg_has(val, 'scheduler'):
                scheduler_dict = val.scheduler.__dict__
                update_schedulers = scheduler_dict.pop('update_schedulers', 'epoch')
                scheduler_name = scheduler_dict.pop('name')

                self.update_schedulers = update_schedulers

                if scheduler_name in ('WarmupMultiStepLR', 'WarmupCosineLR', 'WarmupStepWithFixedGammaLR'):
                    # Get scheduler from fvcore package
                    from fvcore.common.param_scheduler import MultiStepParamScheduler
                    from .lr_scheduler import LRMultiplier, WarmupParamScheduler

                    if scheduler_name == "WarmupMultiStepLR":


                        if verbose and rank() == 0:
                            font = {'color': 'cyan', 'attrs': ('bold', 'dark')}
                            print(pcolor('#' * 100, **font))
                            print(pcolor('#' * 42 + ' Scheduler: WarmupMultiStepLR ' + '#' * 43, **font))
                            print(pcolor('#' * 100, **font))

                        steps = scheduler_dict.pop('steps')
                        max_iter = scheduler_dict.pop('max_iter')
                        gamma = scheduler_dict.pop('gamma')

                        steps = [x for x in steps if x <= max_iter]
                        # if len(steps) != len(steps):
                        #     logger = logging.getLogger(__name__)
                        #     logger.warning(
                        #         "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                        #         "These values will be ignored."
                        #     )
                        sched = MultiStepParamScheduler(
                            values=[gamma**k for k in range(len(steps) + 1)],
                            milestones=steps,
                            num_updates=max_iter,
                        )
                    else:
                        raise NotImplementedError

                    warmup_factor = scheduler_dict.pop('warmup_factor')
                    warmup_iters = scheduler_dict.pop('warmup_iters')
                    warmup_method = scheduler_dict.pop('warmup_method')

                    sched = WarmupParamScheduler(
                        sched,
                        warmup_factor,
                        min(warmup_iters / max_iter, 1.0),
                        warmup_method,
                        rescale_interval=False,
                    )
                    schedulers[key] = LRMultiplier(optimizers[key]['optimizer'], multiplier=sched, max_iter=max_iter)

                else:
                    schedulers[key] = getattr(torch.optim.lr_scheduler, scheduler_name)(**{
                        'optimizer': optimizers[key]['optimizer'],
                        **scheduler_dict,
                    })


        # Return optimizer and scheduler
        return optimizers, schedulers

    def run_arch(self, batch, epoch, flip, unflip, return_logs=False):
        """
        Run model on a batch

        Parameters
        ----------
        batch : Dict
            Dictionary with batch information
        epoch : Int
            Current epoch
        flip : Bool
            Batch should be flipped
        unflip : Bool
            Output should be unflipped
        return_logs: Bool
            If True, return data

        Returns
        -------
        output : Dict
            Dictionary with model outputs
        """
        batch = flip_batch(batch) if flip else batch
        output = self.arch(batch, epoch=epoch, return_logs=return_logs)
        return flip_output(output) if flip and unflip else output

    def training_step(self, batch, epoch, return_logs=False):
        """Processes a training batch"""
        flip_lr = False if self.flip_lr_prob == 0 else \
            random.random() < self.flip_lr_prob

        # TODO(soonminh): draw inputs for debugging
        # from vidar.utils.write import draw_inputs
        # figure = draw_inputs(batch)

        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.run_arch(batch, epoch=epoch, flip=flip_lr, unflip=False, return_logs=return_logs)
        else:
            output = self.run_arch(batch, epoch=epoch, flip=flip_lr, unflip=False, return_logs=return_logs)

        return {
            'metrics': output['metrics'],
            **output,
        }

    def validation_step(self, batch, epoch, return_logs=False):
        """Processes a validation batch"""
        # from vidar.utils.data import break_batch
        # batch = break_batch(batch)

        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.run_arch(batch, epoch=epoch, flip=False, unflip=False)
                flipped_output = None if not self.validate_flipped else \
                    self.run_arch(batch, epoch=epoch, flip=True, unflip=True)
        else:
            output = self.run_arch(batch, epoch=epoch, flip=False, unflip=False, return_logs=return_logs)
            flipped_output = None if not self.validate_flipped else \
                self.run_arch(batch, epoch=epoch, flip=True, unflip=True, return_logs=return_logs)

        if 'batch' in output:
            batch = output['batch']

        results = self.evaluate(batch, output, flipped_output)

        results = [{
            'idx': batch['idx'][i],
            **{key: val[i] for key, val in results['metrics'].items()}
        } for i in range(len(batch['idx']))]

        return output, results

    @staticmethod
    def training_epoch_end():
        """Finishes a training epoch (do nothing for now)"""
        return {}

    def validation_epoch_end(self, output, prefixes):
        """Finishes a validation epoch"""
        if isinstance(output[0], dict):
            output = [output]

        # TODO(soonminh): Create pandas table from output and save into a file

        metrics_dict = {}
        for task in self.metrics:
            metrics_dict.update(
                self.metrics[task].reduce(
                    output, self.datasets['validation'], prefixes))

        return metrics_dict

    def evaluate(self, batch, output, flipped_output=None):
        """
        Evaluate batch to produce predictions and metrics for different tasks

        Parameters
        ----------
        batch : Dict
            Dictionary with batch information
        output : Dict
            Dictionary with output information
        flipped_output : Dict
            Dictionary with flipped output information

        Returns
        -------
        results: Dict
            Dictionary with evaluation results
        """
        # Evaluate different tasks
        metrics, predictions = OrderedDict(), OrderedDict()
        for task in self.metrics:
            task_metrics, task_predictions = \
                self.metrics[task].evaluate(batch, output['predictions'],
                    flipped_output['predictions'] if flipped_output else None)
            metrics.update(task_metrics)
            predictions.update(task_predictions)
        # Crate results dictionary with metrics and predictions
        results = {'metrics': metrics, 'predictions': predictions}
        # Return final results
        return results
