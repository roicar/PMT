'''
    Extract final answer from Qwen output and normalize to label format.
    Usage: python extract.py [dataset_name]
    Default: v1d1C1
'''
import json
import re
import os
import sys

DATASET_NAME = "v3t3E1C1"


# Regex to detect a polynomial-like expression (must contain digit*var or var^N)
# Supports both univariate (x) and multivariate (x0, x1) variable names
POLY_RE = re.compile(r'-?\s*\d+\s*\*?\s*[a-zA-Z]|[a-zA-Z]\w*\s*\^\s*\d+')


def _is_poly(text: str) -> bool:
    '''Check if text looks like a polynomial (has digit*x or x^N).'''
    return POLY_RE.search(text) is not None


# Regex for a full polynomial line (after cleaning): lenient for univariate and multivariate
# Only allows: digits, letters (variables), ^, *, +, -, spaces, parentheses
_FULL_POLY_RE = re.compile(r'^[0-9a-zA-Z\^*+\-\s()]+$')
# Must NOT contain English words (step headers, labels, etc.)
_ENGLISH_RE = re.compile(r'\b[a-zA-Z]{2,}\b')

def _is_full_poly(line: str) -> bool:
    '''Check if the entire line is a polynomial (no extra text).'''
    stripped = line.replace(' ', '')
    if not stripped:
        return False
    # Allow only polynomial-safe characters
    if not _FULL_POLY_RE.match(stripped):
        return False
    # Must not contain English words (e.g. "Step", "Result", "Final")
    if _ENGLISH_RE.search(stripped):
        return False
    return True


def _clean_line(line: str) -> str:
    '''Strip markdown/LaTeX/HTML wrappers and leading '=' from a line.'''
    line = line.strip().strip('*').strip('$')
    line = re.sub(r'<sup>(.*?)</sup>', r'^\1', line)  # <sup>2</sup> -> ^2
    line = re.sub(r'<[^>]+>', '', line)  # strip any remaining HTML tags
    # Python exponent: x**4 -> x^4
    line = line.replace('**', '^')
    # LaTeX subscript: x_0 -> x0, x_1 -> x1
    line = re.sub(r'([a-zA-Z])_(\d+)', r'\1\2', line)
    # LaTeX braces exponent: x^{13} -> x^13
    line = re.sub(r'\^{(\d+)}', r'^\1', line)
    # Insert * between variable groups: x0^13x1^14 -> x0^13*x1^14
    line = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', line)
    line = re.sub(r'^=\s*', '', line)  # 去掉 "= " 前缀
    return line.strip()


def extract_answer(qwen_output: str) -> str:
    '''Extract the final polynomial answer from Qwen's full output.'''
    if not qwen_output:
        return ''

    # Locate the end of Qwen thinking block via regex (robust to tag variants)
    answer_part = qwen_output
    for m in re.finditer(r'<\s*/?\s*think\s*>', qwen_output):
        answer_part = qwen_output[m.end():]

    # Split by '=' and take the last part (handle intermediate steps like "A = B")
    answer_part = answer_part.strip().rstrip('.')
    if '=' in answer_part:
        answer_part = answer_part.split('=')[-1].strip()

    # Strategy 0 (best): Last non-empty line that is a standalone polynomial
    lines = [l for l in answer_part.strip().split('\n') if l.strip()]
    for line in reversed(lines[-5:]):  # check last 5 non-empty lines
        cleaned = _clean_line(line)
        if _is_poly(cleaned) and _is_full_poly(cleaned):
            return cleaned

    # Strategy 1: Bold **text** that is a pure polynomial (skip step headers)
    bold_matches = re.findall(r'\*\*([^*]+?)\*\*', answer_part)
    for m in reversed(bold_matches):
        # Skip if it contains English words (step headers, labels, etc.)
        if re.search(r'[A-Za-z]{2,}', m):
            continue
        cleaned = _clean_line(m)
        if _is_poly(cleaned):
            return cleaned

    # Strategy 2: $polynomial$ (inline LaTeX)
    dollar_matches = re.findall(r'\$([^$]+?)\$', answer_part)
    for m in reversed(dollar_matches):
        cleaned = _clean_line(m)
        if _is_poly(cleaned) and _is_full_poly(cleaned):
            return cleaned

    # Strategy 3 (fallback): search FULL output for last standalone polynomial line
    for line in reversed(qwen_output.strip().split('\n')):
        cleaned = _clean_line(line)
        if _is_poly(cleaned) and _is_full_poly(cleaned):
            return cleaned

    return ''


def normalize_poly(expr: str) -> str:
    '''Normalize a math expression to project label format.

    Examples:
        >>> normalize_poly('7x^2 - 19x + 10')
        '7*x^2-19*x+10'
        >>> normalize_poly('-81x^2 + 45x + 36')
        '-81*x^2+45*x+36'
        >>> normalize_poly('-x + 2')
        '-1*x+2'
    '''
    # Remove all spaces
    s = expr.replace(' ', '')

    # Remove $ and ** wrappers
    s = s.strip('$').strip('*')

    # Strip leading '=' (safety net)
    s = re.sub(r'^=+', '', s)

    # Insert * between digit and letter: 7x -> 7*x
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)

    # Handle implicit coefficient 1: +x -> +1*x, -x -> -1*x, ^x -> ^1*x
    s = re.sub(r'([+\-^])([a-zA-Z])', r'\g<1>1*\2', s)

    # Handle leading variable: x^2 -> 1*x^2, x -> 1*x
    if s and s[0].isalpha():
        s = '1*' + s

    return s


def main():
    dataset = DATASET_NAME

    # deepthink 模式
    # input_file = f'{dataset}_qwen_deepthink_out.jsonl'
    # output_file = f'{dataset}_qwen_deepthink_extracted.jsonl'
 
    # nothink 模式
    input_file = f'{dataset}_qwen_nothink_out.jsonl'
    output_file = f'{dataset}_qwen_nothink_extracted.jsonl'

    # deepseek 模式
    # input_file = f'{dataset}_deepseek_out.jsonl'
    # output_file = f'{dataset}_deepseek_extracted.jsonl'

    if not os.path.exists(input_file):
        print(f'Error: {input_file} not found.')
        return

    correct = 0
    total = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            raw_answer = extract_answer(data.get('qwen', ''))
            normalized = normalize_poly(raw_answer)
            data['predict'] = normalized
            data.pop('qwen', None)

            total += 1
            if normalized == data.get('label', ''):
                correct += 1

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f'Done: {total} samples, {correct} exact matches (string-level)')
    print(f'Output saved to: {output_file}')


if __name__ == '__main__':
    main()