"""Read the first C4o9 samples without changing data, caches or models.

Run from the original project root:
    python unsigned/preview_tokenized.py --n 3
    python unsigned/preview_tokenized.py --raw --n 3
    python unsigned/preview_tokenized.py --path data/accum/C4o9_1000000/data.pt --n 3
"""

import argparse
import inspect
from pathlib import Path
import sys

UNSIGNED_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UNSIGNED_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


VOCAB = ['<pad>', '<s>', '</s>', '>', '+', '-'] + list('0123456789')
FIELDS = (
    'operand', 'operand_id1', 'operand_id2', 'result',
    'result_id1', 'result_id2', 'operand_rmov',
)


def load_data(path):
    # Use the original storage classes when torch unpickles PosCpData.
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(str(path), **options)


def decode(values):
    return ''.join(
        VOCAB[index] if 0 <= index < len(VOCAB) else '<id:{}>'.format(index)
        for index in values.detach().cpu().tolist()
    )


def preview_tokenized(sample, index):
    missing = [name for name in FIELDS if not hasattr(sample, name)]
    if missing:
        raise ValueError(
            'Sample {} is {}, missing fields: {}. Expected C4o9 pc data.'.format(
                index, type(sample).__name__, ', '.join(missing)))
    print('\n' + '=' * 72)
    print('Sample {} | {}.{}'.format(
        index, type(sample).__module__, type(sample).__name__))
    print('Input tokens : {}'.format(decode(sample.operand)))
    print('Target tokens: {}'.format(decode(sample.result)))
    print('Shifted input: {}'.format(decode(sample.operand_rmov)))
    for name in FIELDS:
        value = getattr(sample, name)
        print('{} | shape={} | {} | {}'.format(
            name, tuple(value.shape), value.dtype, value.detach().cpu().tolist()))
    print('Note: result includes <s> and </s>; result PosIDs exclude </s>.')


def preview_raw(sample, index):
    if not isinstance(sample, (list, tuple)) or len(sample) != 2:
        raise ValueError('Sample {} is not an accumulation raw sample.'.format(index))
    operands, scratchpad = sample
    print('\n' + '=' * 72)
    print('Raw sample {}'.format(index))
    print('Operands : {!r}'.format(operands))
    print('Scratchpad: {!r}'.format(scratchpad))
    print('Signed input : {}'.format(''.join(operands)))
    print('Unsigned input: {}'.format(''.join(operands).lstrip('+')))
    print('Unsigned target: {}'.format('>'.join(scratchpad).replace('+', '')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=3, help='number of samples (default: 3)')
    parser.add_argument('--start', type=int, default=0, help='first sample index (default: 0)')
    parser.add_argument('--path', type=Path,
                        help='tokenized data.pt, or raw sample directory with --raw')
    parser.add_argument('--raw', action='store_true', help='inspect numbered raw .pt files')
    args = parser.parse_args()
    if args.n < 1 or args.start < 0:
        parser.error('--n must be positive and --start must be nonnegative')

    if args.raw:
        path = args.path or UNSIGNED_ROOT / 'dataset' / 'accum' / 'C4o9'
    else:
        path = args.path or UNSIGNED_ROOT / 'data' / 'accum' / 'C4o9_1000000' / 'cho' / 'data.pt'
    path = path.expanduser().resolve()
    print('Path: {}'.format(path))
    if not path.exists():
        parser.error('Path does not exist. For unprocessed samples use --raw; '
                     'otherwise finish tokenization or supply --path.')

    if args.raw:
        if not path.is_dir():
            parser.error('--raw requires a directory containing 0.pt, 1.pt, ...')
        for index in range(args.start, args.start + args.n):
            sample_path = path / '{}.pt'.format(index)
            if not sample_path.is_file():
                parser.error('Missing raw sample: {}'.format(sample_path))
            preview_raw(load_data(sample_path), index)
    else:
        if not path.is_file():
            parser.error('Supply a tokenized data.pt file, or use --raw for raw data')
        print('Loading the complete data.pt into CPU memory ...', flush=True)
        data = load_data(path)
        if not isinstance(data, (list, tuple)):
            parser.error('Expected an outer list or tuple, got {}'.format(type(data).__name__))
        print('Container: {} | Samples: {}'.format(type(data).__name__, len(data)))
        if args.start >= len(data):
            parser.error('--start is beyond the available samples')
        for index in range(args.start, min(args.start + args.n, len(data))):
            preview_tokenized(data[index], index)


if __name__ == '__main__':
    main()
