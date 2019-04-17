import json
import gzip
import os


def _convert_ext(filepath, compress):
    b, e = os.path.splitext(filepath)
    filepath = b + '.json'
    if compress:
        filepath += '.gz'
    return filepath


def _check_ext(filepath):
    compress = False
    b, e = os.path.splitext(filepath)
    if e == '.gz':
        compress = True
    return compress


def dump(obj, filepath, compress=None, encoding='utf-8', *args, **kwargs):
    if not compress:
        compress = _check_ext(filepath)
    filepath = _convert_ext(filepath, compress)
    if compress:
        with gzip.GzipFile(filepath, 'w') as f:
            f.write(json.dumps(obj, *args, **kwargs).encode(encoding))
    else:
        with open(filepath, 'w') as f:
            json.dump(obj, f, *args, **kwargs)
    return filepath


def load(filepath, encoding='utf-8', *args, **kwargs):
    compressed = _check_ext(filepath)
    if compressed:
        with gzip.GzipFile(filepath, 'r') as f:
            obj = json.loads(f.read().decode(encoding), *args, **kwargs)
    else:
        with open(filepath, 'r') as f:
            obj = json.load(f, *args, **kwargs)
    return obj
