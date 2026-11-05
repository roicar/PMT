'''
    Utility for tokenization\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 18 Feb 2026
'''
from unsigned.data_utils.args import Arguments
from os.path import join, abspath, dirname, exists
from utils import generate_folder
from typing import List, Union, Tuple, Iterator
from re import sub, match, findall
from data_utils.storage import PosCpData, SequenceData, MulData, AddData, PremData, CharSetData, LongMulData, LongSequenceMulData, PolyGraphSeq
from torch import tensor, load, save, Tensor, where, zeros_like, cat, zeros, ones
from itertools import count
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import HeteroData


class Tokenization(Arguments):
    r'''Class for tokenizing samples.'''

    def __init__(self, **kwargs) -> None:
        r'''Directories and path of files storing data.'''
        super().__init__(**kwargs)
        if not getattr(self, 'collect', False) and getattr(self, 'accum', False) and not getattr(self, 'sum', False):
            self.raw_data_folder = join(
                abspath(dirname(__file__)), '..', 'dataset', 'accum', self.dataset)
            self.processed_data_folder = join(abspath(dirname(
                __file__)), '..', 'data', 'accum', self.dataset + '_' + str(self.dataset_size))
            self.processed_data_folder = join(
                self.processed_data_folder, 'cho' if self.unsigned else 'signed')
        else:
            self.raw_data_folder = join(
                abspath(dirname(__file__)), '..', 'dataset', self.task, self.dataset)
            if self.task == 'mul' and self.dataset != 'v1d15C1':
                self.processed_data_folder = join(abspath(dirname(
                    __file__)), '..', 'data', self.task, self.dataset + '_' + str(self.dataset_size), self.tokenization)
            else:
                self.processed_data_folder = join(abspath(dirname(
                    __file__)), '..', 'data', self.task, self.dataset + '_' + str(self.dataset_size))
            if getattr(self, 'collect', False) + getattr(self, 'combine', False) + getattr(self, 'accum', False) +getattr(self, 'sum', False) == 1:
                subtask_folder = 'collect' if self.collect else 'combine' if self.combine else 'sum'
                self.processed_data_folder = join(
                    self.processed_data_folder, subtask_folder)
            elif self.task == 'prem' and self.dataset not in ['v1d7F7', '_v1D7F7', 'v2t2E1F7', 'v3t3E1F7']:
                if self.finetune:
                    self.processed_data_folder = join(self.processed_data_folder, 'ft')
                else:
                    step_folder = self.prem_step + '-step' if not self.detailed_spa else 'detailed_spa'
                    self.processed_data_folder = join(self.processed_data_folder, step_folder)
            elif self.finetune:
                self.processed_data_folder = join(self.processed_data_folder, 'ft')
        generate_folder(self.processed_data_folder)
        if self.model == 'polynormer':
            self.processed_data_path = join(self.processed_data_folder, 'graph.pt')
        else:
            self.processed_data_path = join(self.processed_data_folder, 'data.pt')

    def tokenize_poly(self, p: str) -> List:
        r'''Tokenize a polynomial.

        Returns a list of separated tokens in the polynomial.

        Args:
            p (str): A polynomial.

        Example (multivariate):
            >>> p = 
            >>> p = 'x2^6+7*x1*x3^54-x0^3-819*x4+63'
            >>> self.tokenize_poly(p) (self.tokenization = 'ImUs')
            ['x2', '^', '6', '+', '7', '*', 'x1', '*', 'x3', '^', '54', '-', 'x0', '^', '3', '-', '819', '*', 'x4', '+', '63']
            >>> self.tokenize_poly(p) (self.tokenization = 'ImDs')
            ['x2', '^', '6', '+', '7', '*', 'x1', '*', 'x3', '^', '54', '-', 'x0', '^', '3', '-', '819', '*', 'x4', '+', '63']

        '''
        if self.tokenization[:2] == 'Ex':
            # Visualize unexplicit '+' sign at the beginning of a polynomial if exists
            p = sub(r'^', '+', p) if not p.startswith('-') else p
            # Visualize unexplicit '1' coefficient
            p = sub(r'(?<=[+-])(?=x\d*)', '1*', p)
            # Visualize unexplicit '1' exponent number
            p = sub(r'(x\d*)(?!\^)(?=(?:[+\-*]|$))', r'\1^1', p)
        p = findall(r'x\d*|\^|[+\-*]|\d+', p)
        # Split digits
        if self.tokenization[-2:] == 'Ds':
            tokens = []
            for token in p:
                if token.isdigit():
                    tokens.extend(list(token))
                else:
                    tokens.append(token)
            return tokens
        return p

    def tokenize_accum_sample(self, sample: List) -> Union[PosCpData, SequenceData]:
        prompt = ''.join(sample[0])
        response = '>'.join(sample[1])
        if self.unsigned:
            prompt = prompt.lstrip('+')
            response = response.replace('+', '')
        # Multi-level position coupling with scratchpad
        # Cho H, Cha J, Bhojanapalli S, et al. Arithmetic Transformers can length-generalize in both operand length and count [C]// Proc of Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Stroudsburg, PA: Association for Computational Linguistics, 2018, 2: 464-468.
        # https://arxiv.org/pdf/2410.15787
        if self.pe_scheme_accum == 'pc':
            # PosID1: digit coupling
            prompt_id1 = tensor(list(map(int, sub(
                r'[+-]|\d+',
                lambda m: '1' if m.group() in '+-'  # PosID1 for '+' or '-': 1
                # PosID1 for digits: e.g. 43 -> [3, 2]
                else ''.join(str(i + 1) for i in range(len(m.group()), 0, -1)),
                prompt))))

            if self.unsigned:
                # Each '+' belongs to the preceding operand in Cho's prompt.
                prompt_id2 = tensor(list(map(int, sub(
                    r'\d+\+?',
                    lambda m, c=count(1): str(next(c)) * len(m.group()),
                    prompt))))

                # '<s>' (the displayed '=') and '>' both use PosID1 1.
                response_id1 = tensor([1] + list(map(int, sub(
                    r'\d+|>',
                    lambda m: '1' if m.group() == '>'
                    else ''.join(str(i + 2) for i in range(len(m.group()))),
                    response))))
                response_id2 = tensor([1] + list(map(int, sub(
                    r'>?\d+',
                    lambda m, c=count(1): (str(next(c)) + ' ') * len(m.group()),
                    response).split())))
            else:
                # PosID2: operand coupling
                prompt_id2 = tensor(list(map(int, sub(
                    r'[+-]?\d+',  # Look for signed number
                    # Each digit (including sign) in n'th operand is emdowed with a PosID2 of n (n=1,2,...)
                    lambda m, c=count(1): str(next(c)) * len(m.group()),
                    prompt))))

                response_id1 = tensor([0] + list(map(int, sub(  # '[0]' ahead denotes PosID1 of '<s>' is 0
                    r'\d+|[+\-]|>',
                    lambda m: ('0' if m.group() == '>'  # Separator for operands ('>') is emdowed with a PosID1 of 0
                               else '1' if m.group() in '+-'
                               # PosID1 for digits in operand with reversed format: e.g. 43 -> '34' -> [2, 3]
                               else ''.join(str(i + 2) for i in range(len(m.group())))),
                    response))))

                response_id2 = tensor([1] + list(map(int, sub(
                    # Operand patterns: <s>D..D<g>, >D..D<g> (<g> in [+-])
                    r'[+-]|>?\d+',
                    lambda m, c=count(1): (
                        str(next(c)) + ' ') * (len(sub(r'\D', '', m.group())) + 1 + ('>' in m.group()))
                    if m.group().lstrip('>+-').isdigit() else '',
                    response).split())))

            prompt_vocabid = tensor([self.vocab_accum.index(
                token) for token in list(prompt) + ['<s>']])
            response_vocabid = tensor([self.vocab_accum.index(token) for token in [
                                      '<s>'] + list(response) + ['</s>']])
            sequence = PosCpData(prompt_vocabid[:-1], prompt_id1, prompt_id2,
                                 response_vocabid, response_id1, response_id2, prompt_vocabid[1:])
        else:
            prompt = list(prompt)
            response = list(response)
            prompt_vocabid = tensor(
                [self.vocab_accum.index(token) for token in list(prompt)])
            response_vocabid = tensor([self.vocab_accum.index(token) for token in [
                                      '<s>'] + list(response) + ['</s>']])
            sequence = SequenceData(prompt_vocabid, response_vocabid)

        return sequence

    def tokenize_simple_arith_sample(self, sample: Union[MulData, AddData, PremData, CharSetData]) -> SequenceData:
        if not getattr(self, 'accum', False) and getattr(self, 'collect', False) ^ getattr(self, 'combine', False):
            if self.task == 'add':
                collect = findall(r'\(|\)|x\d*|\^|[+\-*]|\d+', collect)
                if self.tokenization[-2:] == 'Ds':
                    tokens = []
                    for tok in collect:
                        if tok.isdigit():
                            tokens.extend(list(tok))
                        else:
                            tokens.append(tok)
                    collect = tokens
            else:
                collect = sample.collected_product
                if self.tokenization[:2] == 'Im':
                    collect = sub(r'x\^1(?!\d)', 'x', collect)
                    collect = sub(r'([+-])1\*(?=x)', r'\1', collect)
                elif self.tokenization[-2:] == 'Us':
                    collect = findall(r'\(|\)|x\d*|\^|[+\-*]|\d+', collect)
                else:
                    # Ds: use findall to preserve multi-char tokens (x0, brackets),
                    # then split digits individually
                    collect = findall(r'\(|\)|x\d*|\^|[+\-*]|\d+', collect)
                    tokens = []
                    for tok in collect:
                        if tok.isdigit():
                            tokens.extend(list(tok))
                        else:
                            tokens.append(tok)
                    collect = tokens
        if not getattr(self, 'collect', False) and getattr(self, 'combine', False):
            in_tokens = ['<s>'] + collect + ['</s>']
            out_tokens = ['<s>'] + \
                self.tokenize_poly(sample.product) + ['</s>']
            vocab = self.vocab_collect
        else:
            system = sample.system
            in_tokens = ['<s>'] + self.tokenize_poly(
                system[0]) + ['<op>'] + self.tokenize_poly(system[1]) + ['</s>']
            if getattr(self, 'collect', False) and not getattr(self, 'combine', False) and not getattr(self, 'accum', False):
                out_tokens = ['<s>'] + collect + ['</s>']
                vocab = self.vocab_collect
            else:
                vocab = self.vocab
                if self.task == 'mul':
                    out_tokens = ['<s>'] + \
                        self.tokenize_poly(sample.product) + ['</s>']
                elif self.task == 'add':
                    out_tokens = ['<s>'] + \
                        self.tokenize_poly(sample.sum) + ['</s>']
                elif self.task == 'prem':
                    if self.dataset in ['v1d3C1', 'v1d7C1']:
                        scratchpad = sample.one_spa[-1:] if self.prem_step == '0' else sample.one_spa if self.detailed_spa else sample.one_spa[0::2]
                    elif self.dataset in ['_v1D3C1L3', '_v1D7C1L3', 'v2t2E1C1L3', 'v3t3E1C1L3']:
                        scratchpad = sample.one_spa[-1:] if self.prem_step == '0' else sample.one_spa if self.prem_step == '1' else sample.multi_spa if self.detailed_spa else sample.multi_spa[:1] + [
                            sample.multi_spa[i] for g in range(2, len(sample.multi_spa), 3) for i in (g, g+2)]
                    elif self.dataset in ['v1d7F7', 'v2t2E1F7', 'v3t3E1F7']:
                        scratchpad = sample.one_spa
                    else:
                        scratchpad = sample.multi_spa
                    out_tokens = ['<s>']
                    for res in scratchpad:
                        out_tokens += self.tokenize_poly(res)
                        out_tokens.append('>')
                    out_tokens.pop()
                    out_tokens.append('</s>')
                else:
                    out_tokens = ['<s>']
                    for p in sample.charset:
                        out_tokens += self.tokenize_poly(p) + ['>']
                    out_tokens.pop()
                    out_tokens.append('</s>')
        in_vocabid = tensor([vocab.index(token) for token in in_tokens])
        out_vocabid = tensor([vocab.index(token) for token in out_tokens])
        data_pair = SequenceData(in_vocabid, out_vocabid)
        return data_pair

    def tokenize_long_mul_sample(self, sample: LongMulData) -> Union[List, LongSequenceMulData]:
        if self.collect:
            system = sample.system
            f0, f1, g0, g1 = (tensor([self.vocab.index(token) for token in self.tokenize_poly(
                system[i][j])]) for i in range(2) for j in range(2))
            f0g0, f0g1, f1g0, f1g1 = cat([tensor([1]), f0, tensor([3]), g0, tensor([2])]), cat([tensor([1]), f0, tensor([3]), g1, tensor(
                [2])]), cat([tensor([1]), f1, tensor([3]), g0, tensor([2])]), cat([tensor([1]), f1, tensor([3]), g1, tensor([2])])
            if self.accum and self.sum:
                prod = sample.product
                return LongSequenceMulData(f0g0, f0g1, f1g0, f1g1, prod)
            else:
                collected_prod = sample.collected_product
                p00, p01, p10, p11 = (tensor([self.vocab.index(token) for token in [
                                      '<s>'] + list(prod) + ['</s>']]) for prod in collected_prod)
                return [SequenceData(f0g0, p00), SequenceData(f0g1, p01), SequenceData(f1g0, p10), SequenceData(f1g1, p11)]
        p01_p10, p01_p10_p11, p00_p01_p10_p11 = (tensor([self.vocab.index(token) for token in [
                                                 '<s>'] + self.tokenize_poly(pair[0]) + ['<op>'] + self.tokenize_poly(pair[1]) + ['</s>']]) for pair in sample.combined_product)
        s1, s1_s2, s0_s1_s2 = (tensor([self.vocab.index(token) for token in [
                               '<s>'] + list(_sum) + ['</s>']]) for _sum in sample.collected_sum)
        return [SequenceData(p01_p10, s1), SequenceData(p01_p10_p11, s1_s2), SequenceData(p00_p01_p10_p11, s0_s1_s2)]

    def graphize(self, polys: List, is_yield: bool = False)->Tuple[HeteroData, Tensor]:
        data = HeteroData()
        token = []
        if not is_yield:
            coeff, exp, pow, term, poly = [[] for _ in range(5)]
        u2c_src, u2e_src, u2p_src, e2p_src, c2t_src, p2t_src, t2p_src = [
        [] for _ in range(7)]
        u2c_dst, u2e_dst, u2p_dst, e2p_dst, c2t_dst, p2t_dst, t2p_dst = [
        [] for _ in range(7)]

        for i, p in enumerate(polys):
            p = '+' + p if not p.startswith('-') else p
            p = sub(r'(?<=[+-])(?=x\d*)', '1*', p)
            p = sub(r'(x\d*)(?!\^)(?=(?:[+\-*]|$))', r'\1^1', p)
            tokens = findall(r'x\d*|\^|[+\-*]|\d+', p)
        
            j = 0
            if not is_yield:
                term_start = len(term)
            while j < len(tokens):
                if not is_yield:
                    term_idx = len(term)
                    # Coefficient
                    coeff_idx = len(coeff)
                sign = [tokens[j]]
                unsign = ['c{:s}'.format(c) for c in tokens[j + 1]]
                if is_yield:
                    coeff_tokens = unsign[::-1] + sign
                else:
                    coeff_tokens = sign + unsign
                if not is_yield:
                    u2c_src.extend(
                [len(token) + idx for idx in range(len(coeff_tokens))])
                    u2c_dst.extend([coeff_idx] * len(coeff_tokens))

                coeff_tokens = [self.vocab.index(u) for u in coeff_tokens]
                token.extend(coeff_tokens)
                if not is_yield:
                    coeff.append(coeff_idx)
                j += 2

                # Variable power
                while j < len(tokens) and tokens[j] == '*':
                    if not is_yield:
                        pow_idx = len(pow)
                        # Variable symbol
                        u2p_src.append(len(token))
                        u2p_dst.append(pow_idx)
                    token.append(self.vocab.index(tokens[j + 1]))

                    # Exponent number
                    exp_tokens = ['e{:s}'.format(e) for e in tokens[j + 3]]
                    if is_yield:
                        exp_tokens = exp_tokens[::-1]
                    else:
                        u2e_src.extend(
                    [len(token) + idx for idx in range(len(exp_tokens))])
                        u2e_dst.extend([len(exp)] * len(exp_tokens))

                    exp_tokens = [self.vocab.index(u) for u in exp_tokens]
                    token.extend(exp_tokens)

                    if not is_yield:
                        e2p_src.append(len(exp))
                        e2p_dst.append(pow_idx)
                        exp.append(len(exp))
                        p2t_src.append(pow_idx)
                        p2t_dst.append(term_idx)
                        pow.append(pow_idx)
                    j += 4

                if not is_yield:
                    term_idx = len(term)
                    c2t_src.append(coeff_idx)
                    c2t_dst.append(term_idx)
                    term.append(term_idx)

            if not is_yield:
                t2p_src.extend([idx for idx in range(term_start, len(term))])
                t2p_dst.extend([i] * (len(term) - term_start))
                poly.append(i)

        token = tensor(token)
        if is_yield:
            return token
        data['token'].x = tensor(token).unsqueeze(1)
        data['coeff'].x = tensor(coeff).unsqueeze(1)
        data['exp'].x = tensor(exp).unsqueeze(1)
        data['pow'].x = tensor(pow).unsqueeze(1)
        data['term'].x = tensor(term).unsqueeze(1)
        data['poly'].x = tensor(poly).unsqueeze(1)

        data['token', 'coeff'].edge_index = tensor([u2c_src, u2c_dst])
        data['coeff', 'token'].edge_index = tensor([u2c_dst, u2c_src])
        data['token', 'exp'].edge_index = tensor([u2e_src, u2e_dst])
        data['exp', 'token'].edge_index = tensor([u2e_dst, u2e_src])
        data['token', 'pow'].edge_index = tensor([u2p_src, u2p_dst])
        data['pow', 'token'].edge_index = tensor([u2p_dst, u2p_src])
        data['exp', 'pow'].edge_index = tensor([e2p_src, e2p_dst])
        data['pow', 'exp'].edge_index = tensor([e2p_dst, e2p_src])
        data['coeff', 'term'].edge_index = tensor([c2t_src, c2t_dst])
        data['term', 'coeff'].edge_index = tensor([c2t_dst, c2t_src])
        data['pow', 'term'].edge_index = tensor([p2t_src, p2t_dst])
        data['term', 'pow'].edge_index = tensor([p2t_dst, p2t_src])
        data['term', 'poly'].edge_index = tensor([t2p_src, t2p_dst])
        data['poly', 'term'].edge_index = tensor([t2p_dst, t2p_src])

        return data

    def tokenize(self) -> List:
        if exists(self.processed_data_path):
            if self.new_torch:
                data = load(self.processed_data_path, weights_only=False)
            else:
                data = load(self.processed_data_path)
            print('Tokenized data are loaded from \'{:s}\'.'.format(
                self.processed_data_path))
        else:
            print('Tokenizing data ...')
            data = []
            for i in range(self.dataset_size):
                if self.new_torch:
                    sample = load(join(self.raw_data_folder, str(
                        i) + '.pt'), weights_only=False)
                else:
                    sample = load(join(self.raw_data_folder, str(i) + '.pt'))
                if self.model == 'polynormer':
                    data.append(PolyGraphSeq(self.graphize(sample.system), self.graphize([sample.product], True)))
                elif self.dataset == 'v1d15C1':
                    if self.collect ^ self.sum:
                        data += self.tokenize_long_mul_sample(sample)
                    else:
                        data.append(self.tokenize_long_mul_sample(sample))
                elif not getattr(self, 'collect', False) and getattr(self, 'accum', False) and not getattr(self, 'sum', False):
                    data.append(self.tokenize_accum_sample(sample))
                else:
                    data.append(self.tokenize_simple_arith_sample(sample))
            save(data, self.processed_data_path)
            print('Tokenzied data are saved in \'{:s}\'.'.format(
                self.processed_data_path))
            exit(0)
        return data


def ids_to_str(vocabid: Tensor, vocab: List, return_tokens: bool = False) -> Union[str, Tuple[List, str]]:
    r'''Convert indices in vocabulary to a string.'''
    eoses = where(vocabid == 2)[0]
    eos_pos = eoses[0].item() if len(eoses) else vocabid.size(0)
    vocabid = vocabid[:eos_pos]
    tokens = [vocab[i] for i in vocabid]
    _str = ''.join(tokens)
    if return_tokens:
        return tokens, _str
    return _str

# 修改
def separate_collected_coeffs(products: Tensor, vocab_collect: List, vocab_accum: List, unsigned: bool = False) -> Tuple[List]:
    r'''Recognize all collected coefficients of like terms and convert them into data input to accumulation model.

    Returns lists of replacement positions and vocabulary indices, position ID1's, position ID2's of accumulation formulas.

    Args:
        products (Tensor): Products with coefficients of like terms collected in a batch.
        vocab_collect (Tensor): Vocabulary for the subtask of learning collected product.
        vocab_accum (Tensor): Vocabulary for the task of accumulation.
    '''
    b, m = products.shape
    replacement = zeros_like(products)
    _count = 0
    for i in range(b):
        product = products[i]
        j = 0
        while j < m:
            if product[j] == 4:  # '('
                idx = j + 1
                rbkt_idx = -1
                while idx < m:
                    if product[idx] == 4:  # Encounter new '('
                        j = idx  # Renew recognization of collected coeffcients
                        idx = j + 1
                        continue
                    if product[idx] == 5:  # ')'
                        rbkt_idx = idx
                        break
                    idx += 1
                if rbkt_idx != -1:
                    if rbkt_idx > j + 1:  # ')' must be at the right of '('
                        _count += 1
                        replacement[i, j + 1: rbkt_idx] = _count
                    j = rbkt_idx + 1
                else:
                    break
            else:
                j += 1
    formulas = []
    id1s = []
    id2s = []
    for idx in range(1, _count + 1):
        # Vocabulary indices of tokens in collection of coefficients of a like term
        formula = products[replacement == idx]
        formula = tensor([vocab_accum.index('<pad>' if vid in [
                         3, 4, 5, 8, 9, 10] else vocab_collect[vid]) for vid in formula])
        if unsigned and len(formula) and formula[0] == 4:  # Leading '+'
            formula = formula[1:]
        formulas.append(formula)
        if unsigned:
            # A separator belongs to the operand before it, as in Cho's figure.
            separators = [i + 1 for i, token in enumerate(formula) if token in (4, 5)]
            operands = [formula[s:e] for s, e in zip(
                [0] + separators, separators + [len(formula)])]
            id1 = tensor([position for operand in operands
                          for position in list(range(
                              len(operand) + 1 - int(len(operand) > 0 and operand[-1] in (4, 5)),
                              1, -1)) + ([1] if len(operand) and operand[-1] in (4, 5) else [])])
            id1s.append(id1)
            id2 = tensor([position for position, operand in zip(count(1), operands)
                          for _ in operand])
            id2s.append(id2)
        else:
            # 修改
            # PosID1 (digit position coupling)
            id1 = tensor([id1 for _int in [formula[s:e] for s, e in zip(
                [i for i, t in enumerate(formula) if t in (4, 5)],  # '+', '-'
                # End of an integer
                [i for i, t in enumerate(formula) if t in (4, 5)][1:] + [len(formula)])]
                # PosID1 for ['+', '4', '3'] -> [1, 3, 2]
                for id1 in [1] + list(range(len(_int), 1, -1))])
            id1s.append(id1)
            # PosID2 (operand position coupling)
            id2 = tensor([id2 for id2, _int in zip(count(1), [formula[s:e] for s, e in zip(
                [i for i, t in enumerate(formula) if t in (4, 5)],
                [i for i, t in enumerate(formula) if t in (4, 5)][1:] + [len(formula)])]) for _ in _int])
            id2s.append(id2)
    return replacement, formulas, id1s, id2s


def accum_list_dataloader(formulas: List, id1s: List, id2s: List, batch_size: int, unsigned: bool = False) -> Iterator[Tuple[Tensor]]:
    r'''Construct a dataloader for accumulation of collected coefficients.'''
    for i in range(0, len(formulas), batch_size):
        prompt = pad_sequence(formulas[i: i + batch_size], batch_first=True)
        mask = prompt == 0
        id1 = pad_sequence(id1s[i: i + batch_size], batch_first=True)
        id2 = pad_sequence(id2s[i: i + batch_size], batch_first=True)
        b = prompt.size(0)
        prompt = cat([prompt, ones(b, 1).long()], dim=1)
        mask = cat([mask, zeros(b, 1).bool()], dim=1)
        id1 = cat([id1, (ones if unsigned else zeros)(b, 1).long()], dim=1)
        id2 = cat([id2, ones(b, 1).long()], dim=1)
        yield prompt, mask, id1, id2


def fetch_res_in_scp(scratchpad: List) -> List:
    r'''Parse result from scratchpad.'''
    if '>' not in scratchpad:
        return []
    idx = len(scratchpad) - 1 - scratchpad[::-1].index('>')
    return scratchpad[idx + 1:]
