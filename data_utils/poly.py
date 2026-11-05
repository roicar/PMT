'''
    Utilities for polynomial munipulations\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 28 Jan 2026
'''
from re import compile, fullmatch, findall
from typing import Dict, Union, List, Tuple
from data_utils.int import Int32X, Int32X_accum
from functools import cmp_to_key
from data_utils.token import ids_to_str, fetch_res_in_scp
from torch import Tensor
from os.path import join, abspath, dirname
import platform
from subprocess import check_output
from os import remove


def is_legal_expr(expr: str) -> bool:
    r'''Judges whether an experssion is legal.

    Example for a polynomial with coefficients of like terms collected:
        >>> expr = '+10*x^6(-30+6)*x^5(-20-18-8)*x^4(-25-12+24-16)*x^3(-15+16+48)*x^2(+20+32)*x^1+40'
        >>> print(is_legal_expr(expr))
        True
    '''
    var = r'x\d*'
    # Forbid leading zeros expect single 0
    unsigned = r'(?:0|[1-9][0-9]*)'
    signed = r'(?:[+\-]0|[+\-][1-9][0-9]*)'
    # Power of variable(s) (exponent 1 can be implicit)
    power = fr'{var}(?:\^{unsigned})?'
    # Collection of coefficients embraced by a pair of brackets
    collect = fr'\({signed}(?:{signed})+\)'
    # Power + collection group (e.g. x^6(-30+6))
    powclct = fr'{power}{collect}'
    factor = fr'(?:{unsigned}|{power}|{collect}|{powclct})'
    # Term: one or more factors joined by '*'
    term = fr'{factor}(?:\*{factor})*'
    # Expression: optional leading sign for the first term; subsequent terms joined by +/-
    pattern = fr'^[+\-]?{term}(?:[+\-]{term})*$'
    return compile(pattern).fullmatch(expr) is not None


def parse_term(term: str) -> Dict:
    r'''Parses a term string into a dictionary storing degree, support and coefficient.'''
    coeffs = []
    support = {}  # A list of exponent numbers associated with variables
    deg = 0  # Degree of a term
    for factor in term.split('*'):
        # Coefficient of a regular term
        coeff = fullmatch(r'[+-]?\d+', factor)
        if coeff != None:
            coeff = Int32X(coeff.group())
            # Eliminate current term if 0 is detected in coefficient
            if coeff == 0:
                return None
            # It may contain a product of several integers
            coeffs.append(coeff)
        # Collection of coefficients of a like term
        collect = fullmatch(r'\((?:[+-]\d+){2,}\)', factor)
        if collect:
            coeffs = [Int32X(coeff) for coeff in findall(
                r'[+-]\d+', collect.group()[1:-1])]
            cflag = '+'
        # Power of a variable (exponent number cannot be 0)
        power = fullmatch(r'([+-])?(x\d*?)(?:\^([1-9][0-9]*))?', factor)
        if power:
            sign, var, exp = power.group(1), power.group(2), power.group(3)
            if sign:
                # If there exists an explicit sign, it implies coefficients +1 or -1
                coeffs.append(Int32X('-1' if sign == '-' else '+1'))
            elif len(coeffs) == 0:
                # First factor is a variable without sign/coeff, add implicit 1
                coeffs.append(Int32X('1'))
            var_index = 0 if var == 'x' else int(var[1:])
            # Implicit exponent number: 1
            exp = Int32X('1') if exp is None else Int32X(exp)
            deg += exp
            # It may contain a product of several powers with the same variable base
            if var_index in support.keys():
                support[var_index].append(exp)
            else:
                support[var_index] = [exp]
    if support == {}:
        # No support in a constant term
        if 'cflag' not in locals():
            if len(coeffs) == 1:
                return {'deg': deg, 'coeff': coeffs[0]}
            cflag = '*'
        return {'deg': deg, 'coeff': sorted(coeffs), 'cflag': cflag}
    support = [(var_index, exps[0] if len(exps) == 1 else sorted(exps))
               for var_index, exps in sorted(support.items(), key=lambda power: power[0])]
    if 'cflag' not in locals():
        if coeffs is None:
            # Implicit coefficient: +-1
            return {'deg': deg, 'support': support, 'coeff': Int32X('-1' if term.startswith('-') else '+1')}
        if len(coeffs) == 1:
            return {'deg': deg, 'support': support, 'coeff': coeffs[0]}
        cflag = '*'
    return {'deg': deg, 'support': support, 'coeff': sorted(coeffs), 'cflag': cflag}


def grlex_order(term1: Dict, term2: Dict) -> Union[int, Int32X]:
    r'''Grated-lexicographically orders two terms.'''
    # Rule 1: total degree: bigger first
    d1, d2 = term1['deg'], term2['deg']
    if d1 != d2:
        return d2 - d1
    # Rule 2: exponent numbers by lexicographical orders
    if d1 != 0:
        s1, s2 = term1['support'], term2['support']
        for i in range(min(len(s1), len(s2))):
            x1, x2 = s1[i][0], s2[i][0]
            # Rule 2.1: current variable index: smaller first
            if x1 != x2:
                return x1 - x2
            es1, es2 = s1[i][1], s2[i][1]
            e1, ne1 = (sum(es1), len(es1)) if isinstance(
                es1, List) else (es1, 1)
            e2, ne2 = (sum(es2), len(es2)) if isinstance(
                es2, List) else (es2, 1)
            # Rule 2.2: exponent number of the same variable: bigger first
            if e1 != e2:
                return e2 - e1
            if ne1 == ne2 == 1:
                continue
            # Rule 2.3: addition expression: simpler first
            if ne1 != ne2:
                return ne1 - ne2
            # Rule 2.4: current operand in expression: smaller first
            for j in range(ne1):
                _e1, _e2 = es1[j], es2[j]
                if _e1 != _e2:
                    return _e1 - _e2
    # Rule 3: coefficient
    cs1, cs2 = term1['coeff'], term2['coeff']
    c1_is_formula, c2_is_formula = isinstance(cs1, List), isinstance(cs2, List)
    if not c1_is_formula:
        c1, nc1 = cs1, 1
    elif term1['cflag'] == '+':
        c1, nc1 = sum(cs1), len(cs1)
    else:
        c1, nc1 = Int32X_accum(cs1), len(cs1)
    if not c2_is_formula:
        c2, nc2 = cs2, 1
    elif term2['cflag'] == '+':
        c2, nc2 = sum(cs2), len(cs2)
    else:
        c2, nc2 = Int32X_accum(cs2), len(cs2)
    # Rule 3.1: coefficients of a like term: smaller first
    if c1 != c2:
        return c1 - c2
    # Rule 3.2: addition/multiplication expression: simpler first
    if nc1 != nc2:
        return nc1 - nc2
    # Rule 3.3: current operand in expression: smaller first
    if c1_is_formula and c2_is_formula:
        for i in range(nc1):
            _c1, _c2 = cs1[i], cs2[i]
            if _c1 != _c2:
                return _c1 - _c2
    # Other scenarios
    return 0


def parse_poly(p: str) -> List:
    r'''Parses a polynomial string into a list of term dictionaries.'''
    pattern = r'[+\-]?(?:\([^()]*\)|\d+|x\d*(?:\^\d+)?)(?:\*(?:\([^()]*\)|\d+|x\d*(?:\^\d+)?))*'
    terms = findall(pattern, p)
    p_list = [parse_term(term) for term in terms]
    p_list = [term for term in p_list if term]
    p_list = sorted(p_list, key=cmp_to_key(grlex_order))
    return p_list


def to_simplest_form(p: List) -> List:
    r'''Converts a polynomial in redadunt form to the simplest form.'''
    simpl_p = []
    _p = []
    for term in p:
        support = []
        coeff = term['coeff']
        if 'cflag' in term.keys():
            coeff = sum(coeff) if term['cflag'] == '+' else Int32X_accum(coeff)
            if coeff == 0:
                continue
        if 'support' in term.keys():
            for power in term['support']:
                es = power[1]
                x, e = power[0], sum(es) if isinstance(es, List) else es
                if 'support' not in locals():
                    support = [[x, e]]
                else:
                    support.append([x, e])
        else:
            support = ''
        new_support = True
        for _, _support, _coeff in _p:
            if _support == support:
                new_support = False
                _coeff.append(coeff)
                break
        if new_support:
            _p.append([term['deg'], support, [coeff]])
    for deg, support, coeff in _p:
        coeff = sum(coeff)
        if coeff == 0:
            continue
        if deg == 0:
            simpl_p.append({'deg': deg, 'coeff': coeff})
        else:
            simpl_p.append({'deg': deg, 'support': [tuple(
                power) for power in support], 'coeff': coeff})
    simpl_p = sorted(simpl_p, key=cmp_to_key(grlex_order))
    return simpl_p


def metrics(predict: str, target: str, report_per_term_acc: bool, contain_collected_coeffs: bool = False, is_charset: bool = False) -> Tuple[bool]:
    r'''Count metrics for polynomial arithmetic: whether 
     - (1) the prediction is a legal expression, 
     - (2) the degree of prediction equals to that of target, 
     - (3) the number of terms in simplified prediction equals to that in target, 
     - (4) the support of simplified prediction equals to that of target, 
     - (5) the signed support of simplified prediction equals to that of target, 
     - (6) the leading digit of every coefficient in simplified prediction equals to that in target, 
     - (7) the leading two digits of every coeffcient in simplified prediction equal to those in target, 
     - (8) the predicted polynomial is mathematically equivalent to target polynomial, 
     - (9) the predicted polynomial/set is canonically equivalent to target polynomial.
    '''
    # if is_charset:
    #     LE = CE = ES = False
    #     if target == '<null>':
    #         CSt, ES = [], True
    #     else:
    #         CSt = [parse_poly(pt) for pt in target.split('<osep>')]
    #     CSp = [] if predict == '<null>' else predict.split('<osep>')
    #     if sum([is_legal_expr(poly) for poly in CSp]) == len(CSp):
    #         LE, CSp = True, [parse_poly(pp) for pp in CSp]
    #         CE = CSp == CSt
    #     return LE, CE, ES
    if report_per_term_acc:
        Pt = parse_poly(target)
        count_deg = [0] * 15
        idx_deg = []
        correct = [0] * 15
        for i in range(len(Pt)):
            term = Pt[i]
            deg_val = 0 if term['deg'] == 0 else term['deg'].value
            count_deg[deg_val] += 1
            idx_deg.append((deg_val, i))
    LE = is_legal_expr(predict)
    DE = NE = SPE = SSE = C1E = C2E = ME = CE = False
    if LE:
        if target in ['-0', '+0']:
            if predict in ['-0', '+0']:
                return LE, True, True, True, True, True, True, True, True
            return LE, False, False, False, False, False, False, False, False
        if predict in ['+0', '-0']:
            return LE, False, False, False, False, False, False, False, False
        Pp, Pt = parse_poly(predict), parse_poly(target)
        # 当多项式为空时，返回False，表示不合法表达式
        if not Pp or not Pt:
            return LE, False, False, False, False, False, False, False, False
        CE = Pp == Pt
        if CE:
            if report_per_term_acc:
                return LE, True, True, True, True, True, True, True, CE, count_deg, count_deg
            return LE, True, True, True, True, True, True, True, CE
        Dp, Dt = max(Tp['deg'] for Tp in Pp), max(Tt['deg'] for Tt in Pt)
        DE = Dp == Dt
        if contain_collected_coeffs:
            Pt = to_simplest_form(Pt)
        Sp = to_simplest_form(Pp)
        ME = Sp == Pt
        if ME:
            if report_per_term_acc:
                return LE, DE, True, True, True, True, True, ME, CE, count_deg, count_deg
            return LE, DE, True, True, True, True, True, ME, CE
        NE = len(Sp) == len(Pt)
        if DE and NE:
            SPE = all(Tp['support'] == Tt['support'] if 'support' in Tp.keys() and 'support' in Tt.keys(
            ) else False for Tp, Tt in zip(Sp, Pt) if 'support' in Tp.keys() or 'support' in Tt.keys())
            if SPE:
                if report_per_term_acc:
                    j = 0
                    for deg, idx in idx_deg:
                        for i in range(j, len(Sp)):
                            sp_deg = 0 if Sp[i]['deg'] == 0 else Sp[i]['deg'].value
                            if sp_deg == deg:
                                j = i
                                if Sp[i]['coeff'] == Pt[idx]['coeff']:
                                    correct[deg] += 1
                                break
                SSE = all(Tp['coeff'].sign() == Tt['coeff'].sign()
                          for Tp, Tt in zip(Sp, Pt) if 'support' in Tp.keys())
                if SSE:
                    C1E = all(Tp['coeff'].sigfig(1) == Tt['coeff'].sigfig(1)
                              for Tp, Tt in zip(Sp, Pt))
                    if C1E:
                        C2E = all(Tp['coeff'].sigfig(2) == Tt['coeff'].sigfig(
                            2) for Tp, Tt in zip(Sp, Pt))
        if report_per_term_acc:
            return  LE, DE, NE, SPE, SSE, C1E, C2E, ME, CE, count_deg, correct
    return LE, DE, NE, SPE, SSE, C1E, C2E, ME, CE

class InferenceResults:
    def __init__(self, total: int, vocab: List, task: str, report_per_term:bool=False, contain_collected_coeffs: bool = False):
        self.total = total
        self.vocab = vocab
        self.task = task
        self.contain_collected_coeffs = contain_collected_coeffs
        self.report_per_term = report_per_term
        self.TA = self.SE = self.LE = self.CE = self.ME = self.DE = self.NE = self.SPE = self.SSE = self.C1E = self.C2E = self.nc1 = self.nc2 = self.nc3 = self.nc4 = self.nc5 = self.nc6 = self.nc7 = self.nc8 = self.nc9 = 0
        if report_per_term:
            self.TC = [0] * 15
            self.CC = [0] * 15

    def visualize_error_sample(self, error_type: str, _count: int) -> int:
        print('-' * 64)
        print(error_type)
        print('Input:\n{:s}'.format(
            ids_to_str(self.in_vocabid, self.vocab)))
        print('Target:\n{:s}'.format(self.target))
        if hasattr(self, 'collect_output') and self.collect_output:
            print('Collect output:\n{:s}'.format(self.collect_output))
        print('Prediction:\n{:s}'.format(self.predict))
        _count += 1
        return _count

    def charset_verify(self) -> bool:
        system = ids_to_str(
            self.in_vocabid, self.vocab).__str__().replace('\'', '')
        predict = self.predict.__str__().replace('\'', '')
        maple_script = f'''
read "data_utils/charset_verify.mpl":
print(isCharSet({system}, {predict}));
        '''
        mpl_file_path = join(abspath(dirname(__file__)), 'charset_exam.mpl')
        with open(mpl_file_path, 'w', encoding='utf-8') as file:
            file.write(maple_script)
        try:
            executor = 'cmaple' if platform.system() == 'Windows' else 'maple'
            command = f'{executor} -q {mpl_file_path}'
            output = check_output(command, shell=True, text=True)
            remove(mpl_file_path)
            if output.strip() == 'true':
                return True
        except:
            return False
        return False

    def iter(self, in_vocabid: Tensor, out_vocabid: Union[Tensor, List], predict: Union[Tensor, List]) -> None:
        self.in_vocabid = in_vocabid[1:]
        if isinstance(predict, List):
            p_tokens, self.predict = predict, ''.join(predict)
        else:
            if self.task == 'prem':
                p_tokens, _ = ids_to_str(predict, self.vocab, True)
                # p_tokens = fetch_res_in_scp(p_tokens)
                self.predict = ''.join(p_tokens)
            else:
                p_tokens, self.predict = ids_to_str(predict, self.vocab, True)
        if isinstance(out_vocabid, List):
            t_tokens, self.target = out_vocabid, ''.join(out_vocabid)
        else:
            t_tokens, self.target = ids_to_str(
            out_vocabid[1:], self.vocab, True)
        p_len = len(p_tokens)
        if p_len == len(t_tokens):
            self.TA += sum(1 for j in range(p_len)
                           if p_tokens[j] == t_tokens[j]) / p_len
        _SE = self.predict == self.target
        self.SE += _SE
        if _SE:
            self.LE, self.DE, self.NE, self.SPE, self.SSE, self.C1E, self.C2E, self.ME, self.CE = (
                x + 1 for x in (self.LE, self.DE, self.NE, self.SPE, self.SSE, self.C1E, self.C2E, self.ME, self.CE))
            if self.report_per_term:
                Pt = parse_poly(self.target)
                for term in Pt:
                    deg_val = 0 if term['deg'] == 0 else term['deg'].value
                    self.TC[deg_val] += 1
                    self.CC[deg_val] += 1
        else:
            if self.report_per_term:
                _LE, _DE, _NE, _SPE, _SSE, _C1E, _C2E, _ME, _CE, _TC, _CC = metrics(
                self.predict, self.target, True)
                for i in range(15):
                    self.TC[i] += _TC[i]
                    self.CC[i] += _CC[i]
            else:
                _LE, _DE, _NE, _SPE, _SSE, _C1E, _C2E, _ME, _CE = metrics(
                self.predict, self.target, False, self.contain_collected_coeffs, self.task == 'charset')
            self.LE, self.DE, self.NE, self.SPE, self.SSE, self.C1E, self.C2E, self.ME, self.CE = self.LE + _LE, self.DE + \
                _DE, self.NE + _NE, self.SPE + _SPE, self.SSE + _SSE, self.C1E + \
                _C1E, self.C2E + _C2E, self.ME + _ME, self.CE + _CE
            if not _LE and self.nc1 < 5:
                self.nc1 = self.visualize_error_sample(
                    'Illegal expression', self.nc1)
            elif _CE and not _SE and self.nc2 < 5:
                self.nc2 = self.visualize_error_sample(
                    'Canonical equivalent but different sequence', self.nc2)
            elif not _CE and _ME and self.nc3 < 5:
                self.nc3 = self.visualize_error_sample(
                    'Mathematical equivalent but canonical inequivalent', self.nc3)
            if not _DE and self.nc4 < 5:
                self.nc4 = self.visualize_error_sample(
                    'Different degree', self.nc4)
            elif not _NE and self.nc5 < 5:
                self.nc5 = self.visualize_error_sample(
                    'Different number of terms', self.nc5)
            elif _DE and _NE and not _SPE and self.nc6 < 5:
                self.nc6 = self.visualize_error_sample(
                    'Degree & #Terms correct but support incorrect', self.nc6)
            elif _SPE and not _SSE and self.nc7 < 5:
                self.nc7 = self.visualize_error_sample(
                    'Support correct but signed support incorrect', self.nc7)
            elif _SSE and not _C1E and self.nc8 < 5:
                self.nc8 = self.visualize_error_sample(
                    'Signed support correct but leading digit incorrect', self.nc8)
            elif _C1E and not _C2E and self.nc9 < 5:
                self.nc9 = self.visualize_error_sample(
                    'Leading digit correct but the second digit incorrect', self.nc9)

    def report(self) -> None:
        print('+' * 64)
        print('Metrics')
        print('+' * 64)
        print('Token Acc.: {:.4f}'.format(self.TA / self.total * 100))
        print('Seq Acc.: {:.4f}'.format(self.SE / self.total * 100))
        print('Legal Acc.: {:.4f}'.format(self.LE / self.total * 100))
        print('Degree Acc.: {:.4f}'.format(self.DE / self.total * 100))
        print('#Terms Acc.: {:.4f}'.format(self.NE / self.total * 100))
        print('Support Acc.: {:.4f}'.format(
            self.SPE / self.total * 100))
        print('Signed Support Acc.: {:.4f}'.format(
            self.SSE / self.total * 100))
        print('C1 Acc.: {:.4f}'.format(self.C1E / self.total * 100))
        print('C2 Acc.: {:.4f}'.format(self.C2E / self.total * 100))
        print('Math Acc.: {:.4f}'.format(self.ME / self.total * 100))
        print('Canon. Acc.: {:.4f}'.format(self.CE / self.total * 100))
        if self.report_per_term:
            print('Per-term coefficient Acc.:')
            print([a / b for a, b in zip(self.CC, self.TC)])
