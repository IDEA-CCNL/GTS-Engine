#encoding=utf8

import os
import sys
import importlib

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
model_filenames = [os.path.splitext(os.path.basename(v))[0] for v in scandir(pipeline_dir) if v.endswith('_pipeline.py')]
# import all the model modules
__all__ = [os.path.splitext(os.path.basename(v))[0] for v in scandir(pipeline_dir) if not v.endswith('__init__.py')]
