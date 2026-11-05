'''
    Evaluate extracted LLM predictions using project metrics.
    Usage: python evaluate.py [dataset_name]
    Default: v1d1C1
    Coded and annotated by Bin WANG
    Last modified time: 5 Nov 2026
'''
import json
import sys
import os
from unittest.mock import MagicMock

DATASET_NAME = "v3t3E1C1"

# Mock heavy dependencies to avoid import errors for evaluation only
class _AutoMock(MagicMock):
    '''Module mock that behaves as a module for sys.modules.'''
    __path__ = []
    __file__ = ''

if 'torch_geometric' not in sys.modules:
    _tg = _AutoMock()
    _tg_data = _AutoMock()
    _tg.data = _tg_data
    sys.modules['torch_geometric'] = _tg
    sys.modules['torch_geometric.data'] = _tg_data
if 'utils' not in sys.modules:
    sys.modules['utils'] = _AutoMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils.poly import metrics


METRIC_NAMES = ['LE', 'DE', 'NE', 'SPE', 'SSE', 'C1E', 'C2E', 'MA', 'CA']
METRIC_DESCS = [
    'Legal Expr.',
    'Degree',
    '#Terms',
    'Support',
    'Signed Support',
    'C1 (lead. digit)',
    'C2 (lead. 2 dig.)',
    'Math Acc.',
    'Canon. Acc.',
]


def main():
    dataset = DATASET_NAME

    # input_file = f'{dataset}_qwen_deepthink_extracted.jsonl'
    # log_file = f'{dataset}_qwen_deepthink.log'

    input_file = f'{dataset}_qwen_nothink_extracted.jsonl'
    log_file = f'{dataset}_qwen_nothink.log'

    # input_file = f'{dataset}_deepseek_extracted.jsonl'
    # log_file = f'{dataset}_deepseek.log'

    if not os.path.exists(input_file):
        print(f'Error: {input_file} not found.')
        return

    counts = [0] * 9
    total = 0
    errors = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            predict = data.get('predict', '')
            label = data.get('label', '')
            total += 1

            if not predict:
                errors.append((data['id'], 'empty predict', predict, label))
                continue

            try:
                result = metrics(predict, label, False)
                for i, v in enumerate(result):
                    if v:
                        counts[i] += 1
            except Exception as e:
                errors.append((data['id'], str(e), predict, label))

    # Build report lines
    report = []
    report.append(f'\n--- Evaluation ({input_file}) ---')
    report.append(f'Total: {total} samples')
    report.append('-' * 48)
    for i, name in enumerate(METRIC_NAMES):
        pct = counts[i] / total * 100 if total else 0
        report.append(f'  {METRIC_DESCS[i]:24s} ({name:3s}): {counts[i]:4d} / {total}  ({pct:.2f}%)')
    report.append('-' * 48)


    # Print to console
    for r in report:
        print(r)

    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        for r in report:
            f.write(r + '\n')
    print(f'\n📝 Acc 已追加到: {log_file}')


if __name__ == '__main__':
    main()