# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import math
from collections import defaultdict

import torch
import torch.nn as nn

from vidar.utils.distributed import print0, rank, dist_mode
from vidar.utils.logging import pcolor
from vidar.utils.tensor import same_shape
from vidar.utils.types import is_list


def freeze_layers(network, layers=('ALL',), flag_freeze=True):
    """
    Freeze layers of a network (weights and biases)

    Parameters
    ----------
    network : nn.Module
        Network to be modified
    layers : List or Tuple
        List of layers to freeze/unfreeze ('ALL' for everything)
    flag_freeze : Bool
        Whether the layers will be frozen (True) or not (False)
    """
    if len(layers) > 0:
        for name, parameters in network.named_parameters():
            for layer in layers:
                if layer in name or layer == 'ALL':
                    parameters.requires_grad_(not flag_freeze)


def freeze_norms(network, layers=('ALL',), flag_freeze=True):
    """
    Freeze layers of a network (normalization)

    Parameters
    ----------
    network : nn.Module
        Network to be modified
    layers : List or Tuple
        List of layers to freeze/unfreeze ('ALL' for everything)
    flag_freeze : Bool
        Whether the layers will be frozen (True) or not (False)
    """
    if len(layers) > 0:
        for name, module in network.named_modules():
            for layer in layers:
                if layer in name or layer == 'ALL':
                    if isinstance(module, nn.BatchNorm2d):
                        if hasattr(module, 'weight'):
                            module.weight.requires_grad_(not flag_freeze)
                        if hasattr(module, 'bias'):
                            module.bias.requires_grad_(not flag_freeze)
                        if flag_freeze:
                            module.eval()
                        else:
                            module.train()


def freeze_layers_and_norms(network, layers=('ALL',), flag_freeze=True):
    """Freeze layers and normalizations of a network"""
    freeze_layers(network, layers, flag_freeze)
    freeze_norms(network, layers, flag_freeze)


def make_val_fit(model, key, val, updated_state_dict, strict=False):
    """
    Parse state dictionary to fit a model, and make tensors fit if requested

    Parameters
    ----------
    model : nn.Module
        Network to be used
    key : String
        Which key will be used
    val : torch.Tensor
        Key value
    updated_state_dict : Dict
        Updated dictionary
    strict : Bool
        True if no changes are allowed, False if tensors can be changed to fit

    Returns
    -------
    fit : Int
        Number of tensors that fit the model
    """
    fit = 0
    val_new = model.state_dict()[key]
    if same_shape(val.shape, val_new.shape):
        updated_state_dict[key] = val
        fit += 1
    elif not strict:
        for i in range(val.dim()):
            if val.shape[i] != val_new.shape[i]:
                if val_new.shape[i] > val.shape[i]:
                    ratio = math.ceil(val_new.shape[i] / val.shape[i])
                    val = torch.cat([val] * ratio, i)
                    if val.shape[i] != val_new.shape[i]:
                        val = val[:val_new.shape[i]]
                    if same_shape(val.shape, val_new.shape):
                        updated_state_dict[key] = val
                        fit += 1
                elif val_new.shape[0] < val.shape[i]:
                    val = val[:val_new.shape[i]]
                    if same_shape(val.shape, val_new.shape):
                        updated_state_dict[key] = val
                        fit += 1
    assert fit <= 1  # Each tensor cannot fit 2 or more times
    return fit


def load_checkpoint(model, checkpoint, mappings=None, strict=False, verbose=False, prefix=None, add_prefix=None):
    """
    Load checkpoint into a model

    Parameters
    ----------
    model : nn.Module
        Input network
    checkpoint : String or list[String]
        Checkpoint path (if it's a list, load them in order)
    mappings : list[Tuple]
        Define mappings from checkpoint to model
    strict : Bool
        True if all tensors are required, False if can be partially loaded
    verbose : Bool
        Print information on screen
    prefix : String
        Prefix used to change keys

    Returns
    -------
    model: nn.Module
        Loaded network
    """
    if is_list(checkpoint):
        for ckpt in checkpoint:
            load_checkpoint(model, ckpt, mappings, strict, verbose, prefix, add_prefix)
        return model

    font1 = {'color': 'magenta', 'attrs': ('bold', 'dark')}
    font2 = {'color': 'magenta', 'attrs': ('bold',)}

    if verbose:
        print0(pcolor('#' * 60, **font1))
        print0(pcolor('###### Loading from checkpoint: ', **font1) +
               pcolor('{}'.format(checkpoint), **font2))

    state_dict = torch.load(
        checkpoint,
        map_location='cpu' if dist_mode() == 'cpu' else 'cuda:{}'.format(rank())
    )
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    updated_state_dict = {}

    mappings_dict = defaultdict(list)
    if mappings is not None:
        for key, val in mappings:
            if not isinstance(val, list):
                val = [val]
            mappings_dict[key].extend(val)

    total, fit = len(model.state_dict()), 0
    for key, val in state_dict.items():

        for start in ['model.', 'module.']:
            if key.startswith(start):
                key = key[len(start):]
        if prefix is not None:
            idx = key.find(prefix)
            if idx > -1:
                key = key[(idx + len(prefix) + 1):]

        if add_prefix is not None:
            for k, ap in add_prefix:
                if key.startswith(k):
                    key = ap + key
                    break

        if key in model.state_dict().keys():
            fit += make_val_fit(model, key, val, updated_state_dict, strict=strict)

        for mk, mvlist in mappings_dict.items():
            if mk in key:
                for mv in mvlist:
                    new_key = key.replace(mk, mv)
                    val_new = model.state_dict()[new_key]
                    if same_shape(val.shape, val_new.shape):
                        updated_state_dict[new_key] = val
                        fit += 1

    model.load_state_dict(updated_state_dict, strict=strict)

    if verbose:
        color = 'red' if fit == 0 else 'yellow' if fit < total else 'green'
        print0(pcolor('###### Loaded ', **font1) + \
               pcolor('{}/{}'.format(fit,total), color=color, attrs=('bold',)) + \
               pcolor(' tensors', **font1))
        if fit == 0:
            print0(pcolor('Checkpoints keys: ', **font2))
            print0('\n'.join([k for k in state_dict.keys()][:10]))
            print0(pcolor('Model keys: ', **font1))
            print0('\n'.join([k for k in model.state_dict().keys()][:10]))
        print0(pcolor('#' * 60, **font1))

    return model


def save_checkpoint(filename, wrapper, epoch=None):
    """
    Save checkpoint to disk

    Parameters
    ----------
    filename : String
        Name of the file
    wrapper : nn.Module
        Model wrapper to save
    epoch : Int
        Training epoch
    """
    if epoch is None:
        torch.save({
            'state_dict': wrapper.state_dict(),
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'config': wrapper.cfg,
            'state_dict': wrapper.arch.state_dict(),
        }, filename)
