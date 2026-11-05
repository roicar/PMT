"""
View tokenized collect data: input (f<op>g) and output (collected product).
Usage:
    python preview_tokenized.py --dataset v3d2C1 --n 3
    python preview_tokenized.py --dataset v1d1C1 --n 1 --subtask collect
    python preview_tokenized.py --dataset v3t3E1C1 --n 2
    Coded and annotated by Bin WANG
    Last modified time: 5 Nov 2026
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import torch
from data_utils.args import Arguments
from data_utils.token import ids_to_str

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='v3d2C1')
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--subtask', type=str, default='collect',
                    choices=['collect', 'combine', 'accum', 'sum', 'direct'])
args_pt = parser.parse_args()

# Path to tokenized data
if args_pt.subtask == 'direct':
    data_path = os.path.join('data', 'mul', f'{args_pt.dataset}_100000',
                             'ExDs', 'data.pt')
else:
    data_path = os.path.join('data', 'mul', f'{args_pt.dataset}_100000',
                             'ExDs', args_pt.subtask, 'data.pt')
if not os.path.exists(data_path):
    print(f'File not found: {data_path}')
    print('Available paths:')
    base = os.path.join('data', 'mul', f'{args_pt.dataset}_100000', 'ExDs')
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.endswith('.pt'):
                    print(f'  {os.path.join(root, f)}')
    sys.exit(1)

print(f'Loading: {data_path}')
data = torch.load(data_path, weights_only=False)
print(f'Total samples: {len(data)}')

# Build vocab to decode the tokens
from argparse import Namespace
is_direct = args_pt.subtask == 'direct'
ns = Namespace(
    task='mul', dataset=args_pt.dataset, dataset_size=100000,
    batch_size=128,
    tokenization='ExDs', model='transformer', drop_prob=0.1,
    train=False, valid=True, log=False,
    collect=not is_direct, combine=False, accum=False, sum=False,
    # base model params (needed when collect=False)
    pe_scheme='ape-l', emb_dim=512, num_heads=8, num_ffn_layers=2,
    hidden_dim=2048, activate_fn='relu', norm_scheme='layer',
    num_encoder_layers=6, num_decoder_layers=6,
    # collect model params
    pe_scheme_collect='ape-l', emb_dim_collect=512, num_heads_collect=8,
    num_ffn_layers_collect=2, hidden_dim_collect=2048,
    activate_fn_collect='relu', norm_scheme_collect='layer',
    num_encoder_layers_collect=6, num_decoder_layers_collect=6,
    finetune=False, no_freeze=False,
)
a = Arguments(args=ns)
vocab = a.vocab if is_direct else a.vocab_collect

print(f'Vocab ({len(vocab)} tokens): {vocab}')
print()

for i in range(min(args_pt.n, len(data))):
    d = data[i]
    in_tokens, in_str = ids_to_str(d.in_vocabid, vocab, True)
    out_tokens, out_str = ids_to_str(d.out_vocabid, vocab, True)

    print(f'===== Sample {i} =====')
    print(f'INPUT  ({len(in_tokens):3d} tokens): {in_str}')
    print(f'OUTPUT ({len(out_tokens):3d} tokens): {out_str}')
    print(f'  Token list (first 30): {out_tokens[:30]}')

    # Quick checks
    has_double_plus = out_str.startswith('++')
    has_brackets = '(' in out_tokens and ')' in out_tokens
    starts_with_single_plus = out_str.startswith('+') and not out_str.startswith('++')
    starts_with_minus = out_str.startswith('-')

    flags = []
    if has_double_plus:
        flags.append('DOUBLE ++')
    if starts_with_single_plus:
        flags.append('single +')
    if starts_with_minus:
        flags.append('starts with -')
    if has_brackets:
        flags.append('has ()')
    else:
        # Count multi-coeff groups
        n_multi = sum(1 for t in out_tokens if t == '(')
        if n_multi > 0:
            flags.append(f'has {n_multi} bracket groups')
        else:
            flags.append('NO brackets')

    print(f'  Checks: {", ".join(flags)}')
    print()
