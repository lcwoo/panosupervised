# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from vidar.utils.types import is_seq, is_dict, is_tensor


def iterate1(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var, *args, **kwargs):
        if is_seq(var):
            return [func(v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(var, *args, **kwargs)
    return inner


def iterate2(func):
    """Decorator to iterate over a list (second argument)"""
    def inner(self, var, *args, **kwargs):
        if is_seq(var):
            return [func(self, v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(self, val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(self, var, *args, **kwargs)
    return inner


def iterate12(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var1, var2, *args, **kwargs):
        if is_seq(var1) and is_seq(var2):
            return [func(v1, v2, *args, **kwargs) for v1, v2 in zip(var1, var2)]
        elif is_dict(var1) and is_dict(var2):
            return {key: func(val1, val2, *args, **kwargs)
                    for key, val1, val2 in zip(var1.keys(), var1.values(), var2.values())}
        else:
            return func(var1, var2, *args, **kwargs)
    return inner


def multi_write(func):
    """Decorator to write multiple files"""
    def inner(filename, data, **kwargs):
        if is_seq(data):
            for i in range(len(data)):
                filename_i, ext = filename.split('.')
                filename_i = '%s_%d.%s' % (filename_i, i, ext)
                func(filename_i, data[i], **kwargs)
            return
        elif is_dict(data):
            for key, val in data.items():
                filename_i, ext = filename.split('.')
                filename_i = '%s(%s).%s' % (filename_i, key, ext)
                func(filename_i, val, **kwargs)
            return
        else:
            return func(filename, data)
    return inner


def unravel_output(val, B, N):
    if is_tensor(val):
        return val.view(B, N, *val.shape[1:])
    elif is_seq(val):
        return [unravel_output(v, B, N) for v in val]
    elif is_dict(val):
        return {k: unravel_output(v, B, N) for k, v in val.items()}
    else:
        # Do nothing for other types
        return val


def allow_5dim_tensor(func):
    """Decorator to allow 5-dim input tensors by reshaping, e.g. BxNxCxHxW -> {BxN}xCxHxW"""
    def inner(*args, **kwargs):
        B = N = None
        new_args = []
        for v in args:
            if is_tensor(v) and v.dim() == 5:
                assert (B is None and N is None) or (B == v.shape[0] and N == v.shape[1]), \
                    "All the tensors should have the same values for the first two dimensions(B, N)."
                B, N = v.shape[:2]
                new_args.append(v.view(B * N, *v.shape[2:]))
            else:
                new_args.append(v)

        for k, v in kwargs.items():
            if is_tensor(v) and v.dim() == 5:
                assert (B is None and N is None) or (B == v.shape[0] and N == v.shape[1]), \
                    "All the tensors should have the same values for the first two dimensions(B, N)."
                kwargs[k] = v.view(B * N, *v.shape[2:])

        out = func(*new_args, **kwargs)
        return unravel_output(out, B, N) if B is not None else out
    return inner
