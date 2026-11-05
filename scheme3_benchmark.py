'''
    Evaluate Scheme 3 (Collect-to-Accum) on 100 samples per dataset for Table 6 comparison.

    Usage:
        python scheme3_benchmark.py --dataset DATASET_NAME

    Examples:
        python scheme3_benchmark.py --dataset v1d1C1
        python scheme3_benchmark.py --dataset v1d3C1
        python scheme3_benchmark.py --dataset v1d7C1
        python scheme3_benchmark.py --dataset v2d2C1
        python scheme3_benchmark.py --dataset v2t2E1C1
        python scheme3_benchmark.py --dataset v3t3E1C1
    Coded and annotated by Bin WANG
    Last modified time: 5 Nov 2026
'''
import sys, os, time, json, torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from argparse import Namespace
from data_utils import PolyDataloader
from data_utils import MulData, ids_to_str, separate_collected_coeffs
from data_utils import accum_list_dataloader, simplify_bracketless_collect, fetch_res_in_scp
from data_utils.poly import metrics as poly_metrics
from model_utils.transformer import Transformer
from torch import no_grad, ones, zeros, cat, tensor, where, unique
from torch import load as torch_load, cuda, long
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from os.path import join, dirname, abspath, exists
from typing import List, Tuple, Dict, Any

NUM_SAMPLES = 100
DATASET_SIZE = 100000
START_ID = DATASET_SIZE - NUM_SAMPLES  # 99900


def build_config(dataset: str) -> Dict[str, Any]:
    '''Build configuration dict matching training params for Scheme 3.'''
    # Shared config
    cfg = {
        'task': 'mul',
        'dataset': dataset,
        'dataset_size': DATASET_SIZE,
        'train': False,
        'valid': True,
        'batch_size': 64,
        'tokenization': 'ExDs',
        'model': 'transformer',
        'drop_prob': 0.1,
        'log': False,
        'new_torch': tuple(map(int, torch.__version__.split('+')[0].split('.'))) >= (2, 6),

        # Model architecture (standard: D=512, H=8, NE=6, DE=6)
        'pe_scheme': 'ape-l',
        'emb_dim': 512,
        'num_heads': 8,
        'num_ffn_layers': 2,
        'hidden_dim': 2048,
        'activate_fn': 'relu',
        'norm_scheme': 'layer',
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,

        # Collect sub-task
        'collect': True,
        'pe_scheme_collect': 'ape-l',
        'emb_dim_collect': 512,
        'num_heads_collect': 8,
        'num_ffn_layers_collect': 2,
        'hidden_dim_collect': 2048,
        'activate_fn_collect': 'relu',
        'norm_scheme_collect': 'layer',
        'num_encoder_layers_collect': 6,
        'num_decoder_layers_collect': 6,

        # Accum sub-task
        'accum': True,
        'combine': False,
        'sum': False,
        'pe_scheme_accum': 'pc',
        'emb_dim_accum': 512,
        'num_heads_accum': 8,
        'num_ffn_layers_accum': 2,
        'hidden_dim_accum': 2048,
        'activate_fn_accum': 'relu',
        'norm_scheme_accum': 'layer',
        'num_decoder_layers_accum': 6,

        # Finetune (not used)
        'finetune': False,
        'no_freeze': False,
    }

    # Max sequence lengths per dataset (from args.py)
    seq_lens = {
        'v1d1C1':   (19, 24, 111),
        'v1d3C1':   (43, 84, 111),
        'v1d7C1':   (91, 281, 111),
        'v2d2C1':   (75, 214, 111),
        'v3d2C1':   (139, 606, 111),
        'v2t2E1C1': (43, 54, 111),
        'v3t3E1C1': (87, 164, 111),
    }

    in_len, out_len, accum_len = seq_lens.get(dataset, (87, 164, 111))
    cfg['max_in_seq_len_collect'] = in_len
    cfg['max_out_seq_len_collect'] = out_len
    cfg['max_seq_len_accum'] = accum_len

    return cfg


def tokenize_collect_sample(tokenizer: PolyDataloader, mul_data: MulData) -> tensor:
    '''Tokenize a MulData sample for collect model input.'''
    system = mul_data.system
    in_tokens = (['<s>']
                 + tokenizer.tokenize_poly(system[0])
                 + ['<op>']
                 + tokenizer.tokenize_poly(system[1])
                 + ['</s>'])
    in_vocabid = tensor([tokenizer.vocab_collect.index(tok) for tok in in_tokens], dtype=long)
    return in_vocabid


def run_scheme3_pipeline(
    model_collect: Transformer,
    model_accum: Transformer,
    data_loader: DataLoader,
    device: str,
    max_in_seq_len_collect: int,
    max_out_seq_len_collect: int,
    max_seq_len_accum: int,
    batch_size: int,
    vocab_collect: List,
    vocab_accum: List,
) -> Tuple[int, int, int, float]:
    '''Run collect+accum pipeline, return (correct_MA, correct_CA, total, elapsed).'''
    model_collect.eval()
    model_accum.eval()

    correct_ma = 0
    correct_ca = 0
    total = 0
    t_start = time.time()

    for in_vocabid, in_mask, out_vocabid, labels_str in data_loader:
        b = in_vocabid.size(0)
        # Token IDs are already in vocab_collect space (no +2 mapping needed)
        in_mask = in_mask.to(device)

        # ---- Step 1: Generate collected product (autoregressive) ----
        with no_grad():
            nid_collect = model_collect(
                ones(b).long().to(device), None,
                in_vocabid.to(device), in_mask,
                max_seq_len=[max_in_seq_len_collect, max_out_seq_len_collect]
            ).argmax(dim=-1)
        predicts_collect = nid_collect.unsqueeze(1).cpu()

        for i in range(1, max_out_seq_len_collect - 1):
            done = (predicts_collect == 2).any(dim=1).sum().item()
            if done == b:
                break
            with no_grad():
                nid_collect = model_collect(nid_collect, in_mask=in_mask).argmax(dim=-1)
            predicts_collect = cat([predicts_collect, nid_collect.unsqueeze(1).cpu()], dim=1)

        predicts_collect = predicts_collect.to('cpu')

        # ---- Step 2: Simplify collected product ----
        has_brackets = (predicts_collect == 4).any().item()

        if has_brackets:
            # Bracket-based: extract coefficients → accum model
            replacement, prompts, prompt_id1s, prompt_id2s = separate_collected_coeffs(
                predicts_collect, vocab_collect, vocab_accum)

            combined_coeffs = []
            for prompt, mask, prompt_id1, prompt_id2 in accum_list_dataloader(
                prompts, prompt_id1s, prompt_id2s, batch_size
            ):
                b_accum = prompt.size(0)
                nid1 = zeros(b_accum).long().to(device)
                nid2 = ones(b_accum).long().to(device)

                for j in range(max_seq_len_accum - prompt.size(1)):
                    if j:
                        with no_grad():
                            nid_accum = model_accum(
                                x=nid_accum.to(device), x_mask=mask.to(device),
                                posid1=tensor(nid1).to(device),
                                posid2=tensor(nid2).to(device)
                            ).argmax(dim=-1)
                        predicts_accum = cat([predicts_accum, nid_accum.unsqueeze(1)], dim=1)
                    else:
                        with no_grad():
                            nid_accum = model_accum(
                                x=prompt.to(device), x_mask=mask.to(device),
                                posid1=prompt_id1.to(device),
                                posid2=prompt_id2.to(device),
                                max_seq_len=max_seq_len_accum
                            ).argmax(dim=-1)
                        predicts_accum = nid_accum.unsqueeze(1)

                    if (predicts_accum == 2).any(dim=1).sum() == b_accum:
                        break

                    nid1[where(nid1 == 0)[0]] = 1
                    nid1 += 1
                    nid1[where((nid_accum == 4) | (nid_accum == 5))[0]] = 1
                    nid1[where((nid_accum < 4) | (nid1 > 33))[0]] = 0
                    nid2 += nid_accum == 3
                    nid2[where((nid_accum < 3) | (nid2 > 33))[0]] = 0
                    mask = cat([mask, zeros(b_accum, 1).bool()], dim=1)

                predicts_accum = predicts_accum.to('cpu')
                for k in range(b_accum):
                    combined_coeffs.append(fetch_res_in_scp(
                        ids_to_str(predicts_accum[k], vocab_accum))[::-1])

            # Replace coefficients in collect output
            for i in range(b):
                p_tokens, _ = ids_to_str(predicts_collect[i], vocab_collect, True)
                replaces = tensor(replacement[i])
                for rid in unique(replaces[replaces > 0]).sort(descending=True).values:
                    collect_idx = (replaces == rid).nonzero(as_tuple=True)[0]
                    lbound, rbound = collect_idx.min(), collect_idx.max()
                    p_tokens[lbound - 1: rbound + 2] = list(combined_coeffs[rid - 1])
                predict_str = ''.join(p_tokens)
                total += 1
                try:
                    result = poly_metrics(predict_str, labels_str[i], False)
                    if result[7]:   # MA
                        correct_ma += 1
                    if result[8]:   # CA
                        correct_ca += 1
                except Exception:
                    pass

        else:
            # Bracketless: simplify directly
            for i in range(b):
                p_tokens = simplify_bracketless_collect(predicts_collect[i], vocab_collect)
                predict_str = ''.join(p_tokens)
                total += 1
                try:
                    result = poly_metrics(predict_str, labels_str[i], False)
                    if result[7]:   # MA
                        correct_ma += 1
                    if result[8]:   # CA
                        correct_ca += 1
                except Exception:
                    pass

    elapsed = time.time() - t_start
    return correct_ma, correct_ca, total, elapsed


class CollateWithLabels:
    '''Collation that preserves label strings.'''
    def __init__(self, labels: List[str]):
        self.labels = labels

    def __call__(self, batch):
        in_seqs, out_seqs, idxs = zip(*batch)
        in_padded = pad_sequence(in_seqs, batch_first=True)
        in_mask = in_padded == 0
        out_padded = pad_sequence(out_seqs, batch_first=True)
        return in_padded, in_mask, out_padded, [self.labels[i] for i in idxs]


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_vocabid, out_seed = self.data[idx]
        return in_vocabid, out_seed, idx


def main():
    # Parse --dataset from command line
    dataset = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--dataset' and i < len(sys.argv):
            dataset = sys.argv[i + 1]
            break

    if dataset is None:
        print('Usage: python scheme3_benchmark.py --dataset DATASET_NAME')
        print('Example: python scheme3_benchmark.py --dataset v1d1C1')
        print('Available: v1d1C1, v1d3C1, v1d7C1, v2d2C1, v2t2E1C1, v3t3E1C1')
        sys.exit(1)

    # Build config
    cfg = build_config(dataset)
    args = Namespace(**cfg)

    # Create PolyDataloader instance (inherits Tokenization → Arguments, provides tokenize_poly + vocab)
    print(f'Initializing PolyDataloader (dataset={dataset}, tokenization=ExDs) ...')
    tokenizer = PolyDataloader(args=args)

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ---- Build models ----
    print('Building collect model ...')
    model_collect = Transformer(
        tokenizer.pe_scheme_collect, len(tokenizer.vocab_collect),
        tokenizer.emb_dim_collect, tokenizer.num_heads_collect,
        tokenizer.num_ffn_layers_collect, tokenizer.hidden_dim_collect,
        tokenizer.activate_fn_collect, tokenizer.norm_scheme_collect,
        tokenizer.drop_prob, tokenizer.num_encoder_layers_collect,
        tokenizer.num_decoder_layers_collect
    ).to(device)

    print('Building accum model (decoder-only, pc position embedding) ...')
    model_accum = Transformer(
        tokenizer.pe_scheme_accum, len(tokenizer.vocab_accum),
        tokenizer.emb_dim_accum, tokenizer.num_heads_accum,
        tokenizer.num_ffn_layers_accum, tokenizer.hidden_dim_accum,
        tokenizer.activate_fn_accum, tokenizer.norm_scheme_accum,
        tokenizer.drop_prob, 0, tokenizer.num_decoder_layers_accum
    ).to(device)

    # ---- Load model weights ----
    base_model_dir = "/home/xst/Desktop/PolyMul/model"
    new_torch = cfg['new_torch']

    collect_model_name = (f'PE_{tokenizer.pe_scheme_collect}_D_{tokenizer.emb_dim_collect}_'
                          f'H_{tokenizer.num_heads_collect}_NE_{tokenizer.num_encoder_layers_collect}_'
                          f'DE_{tokenizer.num_decoder_layers_collect}.pth')
    collect_model_path = join(base_model_dir, 'mul',
                              f'{dataset}_{DATASET_SIZE}', 'ExDs', 'collect', collect_model_name)

    accum_model_name = (f'PE_{tokenizer.pe_scheme_accum}_D_{tokenizer.emb_dim_accum}_'
                        f'H_{tokenizer.num_heads_accum}_NE_0_DE_{tokenizer.num_decoder_layers_accum}.pth')
    accum_model_path = join(base_model_dir, 'accum', 'C4o9_1000000', accum_model_name)

    if not exists(collect_model_path):
        print(f'\nERROR: Collect model not found!')
        print(f'  Path: {collect_model_path}')
        print(f'\nSkipping {dataset} — no collect model checkpoint available.\n')
        sys.exit(1)

    if not exists(accum_model_path):
        print(f'\nERROR: Accum model not found!')
        print(f'  Path: {accum_model_path}')
        sys.exit(1)

    print(f'Loading collect model: {collect_model_path}')
    sd = torch_load(collect_model_path, weights_only=False) if new_torch else torch_load(collect_model_path)
    model_collect.load_state_dict(sd)

    print(f'Loading accum model: {accum_model_path}')
    sd = torch_load(accum_model_path, weights_only=False) if new_torch else torch_load(accum_model_path)
    model_accum.load_state_dict(sd)

    # ---- Load and tokenize 100 samples ----
    raw_data_folder = join(dirname(dirname(__file__)), 'dataset', 'mul', dataset)
    print(f'\nLoading {NUM_SAMPLES} samples from {raw_data_folder} '
          f'(IDs {START_ID}–{DATASET_SIZE - 1}) ...')

    tokenized_data = []
    labels = []
    for idx in range(START_ID, DATASET_SIZE):
        pt_path = join(raw_data_folder, f'{idx}.pt')
        if not exists(pt_path):
            print(f'  WARNING: {pt_path} not found, skipping')
            continue
        sample = torch_load(pt_path, weights_only=False)

        # Convert loaded dict-like object to MulData
        mul_data = MulData(
            system=sample.system,
            product=sample.product,
            collected_product=getattr(sample, 'collected_product', None),
        )

        in_vocabid = tokenize_collect_sample(tokenizer, mul_data)
        tokenized_data.append((in_vocabid, tensor([0], dtype=long)))
        labels.append(mul_data.product)

    total_samples = len(tokenized_data)
    print(f'  Successfully tokenized {total_samples} samples.')

    if total_samples == 0:
        print('ERROR: No samples could be loaded!')
        sys.exit(1)

    # ---- Create DataLoader ----
    collator = CollateWithLabels(labels)
    ds = SimpleDataset(tokenized_data)
    batch_sz = min(64, total_samples)
    loader = DataLoader(ds, batch_size=batch_sz, shuffle=False, collate_fn=collator)

    # ---- Run pipeline ----
    print(f'\nRunning Scheme 3 (Collect → Accum) on {total_samples} samples ...')
    correct_ma, correct_ca, total, elapsed = run_scheme3_pipeline(
        model_collect, model_accum, loader, device,
        tokenizer.max_in_seq_len_collect,
        tokenizer.max_out_seq_len_collect,
        tokenizer.max_seq_len_accum,
        batch_sz,
        tokenizer.vocab_collect,
        tokenizer.vocab_accum,
    )

    # ---- Report ----
    ma_pct = correct_ma / total * 100 if total else 0.0
    ca_pct = correct_ca / total * 100 if total else 0.0

    # Build result JSON
    result = {
        'dataset': dataset,
        'scheme': 'Scheme 3 (Collect-to-Accum)',
        'samples': total,
        'MA': round(ma_pct, 2),
        'CA': round(ca_pct, 2),
        'Total_Time_s': round(elapsed, 2),
    }

    # Build report text
    report_lines = []
    report_lines.append('')
    report_lines.append('=' * 60)
    report_lines.append(f'  Scheme 3 (Collect-to-Accum) on {dataset} ({total} samples)')
    report_lines.append('=' * 60)
    report_lines.append(f'  Math Acc. (MA):  {correct_ma:4d} / {total}  ({ma_pct:.2f}%)')
    report_lines.append(f'  Canon. Acc. (CA):{correct_ca:4d} / {total}  ({ca_pct:.2f}%)')
    report_lines.append(f'  Total Time:      {elapsed:.2f}s')
    if total:
        report_lines.append(f'  Avg Time/Sample: {elapsed / total:.4f}s')
    report_lines.append('=' * 60)
    report_lines.append(f'\n{json.dumps(result, ensure_ascii=False, indent=2)}')
    report_lines.append('')

    # Print to console
    for line in report_lines:
        print(line)

    # Save to log file
    log_path = f'{dataset}_scheme3.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        for line in report_lines:
            f.write(line + '\n')
    print(f'Results saved to: {os.path.abspath(log_path)}')


if __name__ == '__main__':
    main()
