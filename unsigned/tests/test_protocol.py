"""Check the minimal unsigned changes against original source and real tensors.

These checks compile only selected source AST nodes.  They neither import the
project's package initializers nor replace any production dependency or runner.
No data generation, model construction, training or official evaluation runs.
"""

import __future__
import ast
import hashlib
from itertools import count
import json
from pathlib import Path
from re import findall, match, sub
from types import SimpleNamespace
import unittest

import torch
from torch.nn.utils.rnn import pad_sequence


UNSIGNED_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = UNSIGNED_ROOT.parent
ACCUM_VOCAB = ['<pad>', '<s>', '</s>', '>', '+', '-'] + list('0123456789')
COLLECT_VOCAB = ['<pad>', '<s>', '</s>', '<op>', '(', ')', '+', '-', '*', '^', 'x'] + list('0123456789')


def _tree(path):
    return ast.parse(Path(path).read_text(encoding='utf-8'))


def _definitions(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[node.name + '.' + child.name] = child
    return result


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _execute_nodes(nodes, namespace, filename):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(filename), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)


def _token_functions(path):
    """Load actual selected functions with their actual torch operations."""
    namespace = {
        'tensor': torch.tensor, 'where': torch.where, 'zeros_like': torch.zeros_like,
        'cat': torch.cat, 'ones': torch.ones, 'zeros': torch.zeros,
        'pad_sequence': pad_sequence, 'count': count, 'sub': sub,
        'findall': findall, 'match': match,
    }
    storage_path = PROJECT_ROOT / 'data_utils' / 'storage.py'
    storage_nodes = [node for node in _tree(storage_path).body
                     if isinstance(node, ast.ClassDef) and node.name in {'PosCpData', 'SequenceData'}]
    _execute_nodes(storage_nodes, namespace, storage_path)
    definitions = _definitions(_tree(path))
    wanted = ['Tokenization.tokenize_accum_sample', 'Tokenization.tokenize_poly',
              'ids_to_str', 'fetch_res_in_scp', 'separate_collected_coeffs', 'accum_list_dataloader']
    _execute_nodes([definitions[name] for name in wanted], namespace, path)
    return namespace


class OriginalSourceTests(unittest.TestCase):
    def test_all_44_original_source_hashes_are_unchanged(self):
        baseline = json.loads((UNSIGNED_ROOT / 'ORIGINAL_SOURCES.json').read_text(encoding='utf-8'))
        self.assertEqual(len(baseline), 44)
        for relative, expected in baseline.items():
            with self.subTest(path=relative):
                path = PROJECT_ROOT.joinpath(*relative.replace('\\', '/').split('/'))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_dataloader_is_original_except_for_imports(self):
        original = _tree(PROJECT_ROOT / 'data_utils' / 'dataloader.py')
        local = _tree(UNSIGNED_ROOT / 'data_utils' / 'dataloader.py')
        for module in [original, local]:
            module.body = [node for node in module.body
                           if not isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(_dump(local), _dump(original))

    def test_runner_training_and_run_are_original(self):
        before = _definitions(_tree(PROJECT_ROOT / 'model_utils' / 'inference.py'))
        after = _definitions(_tree(UNSIGNED_ROOT / 'model_utils' / 'inference.py'))
        for name in ['Runner.training', 'Runner.train_polynormer', 'Runner.run']:
            with self.subTest(method=name):
                self.assertEqual(_dump(after[name]), _dump(before[name]))

    def test_unmodified_token_methods_are_original(self):
        before = _definitions(_tree(PROJECT_ROOT / 'data_utils' / 'token.py'))
        after = _definitions(_tree(UNSIGNED_ROOT / 'data_utils' / 'token.py'))
        for name in ['ids_to_str', 'fetch_res_in_scp', 'Tokenization.tokenize_poly',
                     'Tokenization.tokenize_simple_arith_sample',
                     'Tokenization.tokenize_long_mul_sample', 'Tokenization.graphize',
                     'Tokenization.tokenize']:
            with self.subTest(method=name):
                self.assertEqual(_dump(after[name]), _dump(before[name]))


class TensorProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = _token_functions(PROJECT_ROOT / 'data_utils' / 'token.py')
        cls.local = _token_functions(UNSIGNED_ROOT / 'data_utils' / 'token.py')

    def _tokenize(self, sample, unsigned=False, original=False):
        state = SimpleNamespace(unsigned=unsigned, pe_scheme_accum='pc', vocab_accum=ACCUM_VOCAB)
        functions = self.original if original else self.local
        return functions['tokenize_accum_sample'](state, sample)

    def test_signed_tensor_fields_equal_original(self):
        samples = [
            [['+28', '+96', '+77'], ['0+', '82+', '421+', '102+']],
            [['+43', '-50', '+7', '+20'], ['0+', '34+', '7-', '0+', '02+']],
        ]
        for sample in samples:
            before = self._tokenize(sample, original=True)
            after = self._tokenize(sample)
            self.assertEqual(set(vars(after)), set(vars(before)))
            for name in vars(before):
                with self.subTest(sample=sample, field=name):
                    self.assertTrue(torch.equal(getattr(after, name), getattr(before, name)))
                    self.assertEqual(getattr(after, name).dtype, getattr(before, name).dtype)

    def test_cho_complete_figure_tokens_and_position_ids(self):
        sample = [['+28', '+96', '+77'], ['0+', '82+', '421+', '102+']]
        sequence = self._tokenize(sample, unsigned=True)
        prompt = self.local['ids_to_str'](sequence.operand, ACCUM_VOCAB)
        response = self.local['ids_to_str'](sequence.result, ACCUM_VOCAB)
        self.assertEqual(prompt, '28+96+77')
        self.assertEqual(response, '<s>0>82>421>102')
        self.assertEqual(prompt + response.replace('<s>', '='), '28+96+77=0>82>421>102')
        full1 = torch.cat([sequence.operand_id1, sequence.result_id1]).tolist()
        full2 = torch.cat([sequence.operand_id2, sequence.result_id2]).tolist()
        self.assertEqual(full1, [3, 2, 1, 3, 2, 1, 3, 2, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(full2, [1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        self.assertEqual(len(sequence.operand_id1), len(sequence.operand))
        self.assertEqual(len(sequence.result_id1), len(sequence.result) - 1)
        self.assertEqual(sequence.result[-1].item(), 2)
        self.assertTrue(torch.equal(sequence.operand_rmov,
                                    torch.cat([sequence.operand[1:], torch.tensor([1])])))

    def test_collected_prompt_and_bos_positions_equal_training(self):
        text = '+6*x^2(+28+96+77)*x^1+20'
        products = torch.tensor([[COLLECT_VOCAB.index(token) for token in text] + [2]])
        replacement, formulas, id1s, id2s = self.local['separate_collected_coeffs'](
            products, COLLECT_VOCAB, ACCUM_VOCAB, unsigned=True)
        self.assertEqual(len(formulas), 1)
        self.assertEqual(self.local['ids_to_str'](formulas[0], ACCUM_VOCAB), '28+96+77')
        # Replacement indices still cover the original signed-looking bracket
        # contents; only the input sent to the unsigned model loses its first +.
        covered = ''.join(COLLECT_VOCAB[index] for index in products[replacement == 1].tolist())
        self.assertEqual(covered, '+28+96+77')
        batch = list(self.local['accum_list_dataloader'](formulas, id1s, id2s, 1, unsigned=True))
        prompt, mask, id1, id2 = batch[0]
        training = self._tokenize([['+28', '+96', '+77'], ['0+', '82+', '421+', '102+']], unsigned=True)
        self.assertTrue(torch.equal(prompt[0], torch.cat([training.operand, torch.tensor([1])])))
        self.assertTrue(torch.equal(id1[0], torch.cat([training.operand_id1, training.result_id1[:1]])))
        self.assertTrue(torch.equal(id2[0], torch.cat([training.operand_id2, training.result_id2[:1]])))
        self.assertFalse(mask.any().item())

    def test_signed_collected_helper_and_bos_equal_original(self):
        text = '+6*x^2(-28+96+77)*x^1+20'
        products = torch.tensor([[COLLECT_VOCAB.index(token) for token in text] + [2]])
        before = self.original['separate_collected_coeffs'](products, COLLECT_VOCAB, ACCUM_VOCAB)
        after = self.local['separate_collected_coeffs'](products, COLLECT_VOCAB, ACCUM_VOCAB)
        self.assertTrue(torch.equal(before[0], after[0]))
        for left, right in zip(before[1:], after[1:]):
            for left_tensor, right_tensor in zip(left, right):
                self.assertTrue(torch.equal(left_tensor, right_tensor))
        before_batch = next(self.original['accum_list_dataloader'](*before[1:], 1))
        after_batch = next(self.local['accum_list_dataloader'](*after[1:], 1))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(before_batch, after_batch)))
        self.assertEqual(after_batch[2][0, -1].item(), 0)

    def test_missing_eos_and_last_result_parsing_are_original(self):
        text = '0>82>421>102'
        ids = torch.tensor([ACCUM_VOCAB.index(token) for token in text])
        for source in [self.original, self.local]:
            self.assertEqual(source['ids_to_str'](ids, ACCUM_VOCAB), text)
            self.assertEqual(source['fetch_res_in_scp'](text)[::-1], '201')
            self.assertEqual(source['fetch_res_in_scp']('102'), [])
        with_eos_and_junk = torch.cat([ids, torch.tensor([2, 0, 1])])
        self.assertEqual(self.local['ids_to_str'](with_eos_and_junk, ACCUM_VOCAB), text)

    def test_runner_coefficient_fill_keeps_predicted_leading_zeros(self):
        runner_path = UNSIGNED_ROOT / 'model_utils' / 'inference.py'
        definitions = _definitions(_tree(runner_path))
        for method_name in ['Runner.collect_and_accum', 'Runner.collect_and_accum_submodule']:
            method = definitions[method_name]
            loop = next(node for node in ast.walk(method)
                        if isinstance(node, ast.For) and node.body
                        and isinstance(node.body[0], ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'combined_coeff'
                                for target in node.body[0].targets))
            for text, expected in [('001', []), ('0>', ''), ('0>100', '+001'),
                                   ('0>3-', '-3'), ('0>3+', '+3')]:
                predictions = torch.tensor([[ACCUM_VOCAB.index(token) for token in text] + [2]])
                namespace = dict(self.local, b_accum=1, predicts_accum=predictions,
                                 self=SimpleNamespace(unsigned=True, vocab_accum=ACCUM_VOCAB),
                                 combined_coeffs=[])
                _execute_nodes([loop], namespace, runner_path)
                with self.subTest(method=method_name, scratchpad=text):
                    self.assertEqual(namespace['combined_coeffs'], [expected])
                    if expected == '+001':
                        self.assertEqual('+6*x^2' + namespace['combined_coeffs'][0] + '*x^1+20',
                                         '+6*x^2+001*x^1+20')


if __name__ == '__main__':
    unittest.main()
