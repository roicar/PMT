'''
    Generate datasets of integer accumulation\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 12 Feb 2026
'''
from os.path import join, abspath, dirname
import sys  # noqa
sys.path.append(join(abspath(dirname(__file__)), '..', '..', '..'))  # noqa
from utils.folder import generate_folder
from numpy.random import seed, randint
from numpy import concatenate, arange
from torch import save
from argparse import ArgumentParser


def run(N: int, C: int, o: int) -> None:
    r'''
        Generate integer accumulation samples.

        Cho H, Cha J, Bhojanapalli S, et al. Arithmetic Transformers can length-generalize in both operand length and count [C]// Proc of Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Stroudsburg, PA: Association for Computational Linguistics, 2018, 2: 464-468.
        https://arxiv.org/pdf/2410.15787

        Args:
            N (int): Dataset size.
            C (int): The maximum number of digit(s) in an integer.
            o (int): The maximum number of operands.
    '''
    base_dir = join(abspath(dirname(__file__)), '..', '..',
                    'dataset', 'accum', 'C' + str(C) + 'o' + str(o))
    generate_folder(base_dir)
    seed(0)
    ranges = [arange(10 ** (c - 1), 10 ** c) for c in range(1, C + 1)]
    unqiue = []  # Record unique integer accumulation formulas
    for i in range(0, N, 2):
        # The first chunck (variable number of digits)
        while True:
            _o = randint(2, o + 1)
            ints, operands, _sum, scratchpad = [], [], 0, ['0+']
            for _ in range(_o):
                c = randint(1, C + 1)
                _int = ranges[c - 1][randint(0, 9 * 10 ** (c - 1))]
                ints.append(_int)
                operands.append(str(_int) if _int < 0 else '+' + str(_int))
                _sum += _int
                scratchpad.append(str(_sum)[::-1] + ('' if _sum < 0 else '+'))
            ints = ''.join([str(_int) for _int in sorted(ints)])
            if ints not in unqiue:
                unqiue.append(ints)
                break
        save([operands, scratchpad], join(base_dir, str(i) + '.pt'))
        # The second chunck (fixed number of digits)
        while True:
            _o = randint(2, o + 1)
            ints, operands, _sum, scratchpad = [], [], 0, ['0+']
            c = randint(1, C + 1)
            for _ in range(_o):
                _int = ranges[c - 1][randint(0, 9 * 10 ** (c - 1))]
                ints.append(_int)
                operands.append(str(_int) if _int < 0 else '+' + str(_int))
                _sum += _int
                scratchpad.append(str(_sum)[::-1] + ('' if _sum < 0 else '+'))
            ints = ''.join([str(_int) for _int in sorted(ints)])
            if ints not in unqiue:
                unqiue.append(ints)
                break
        save([operands, scratchpad], join(base_dir, str(i + 1) + '.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--N', type=int, default=1000000,
                        help='dataset size (default: 1000000)')
    parser.add_argument('--C', type=int, default=4,
                        help='the maximum number of digit(s) in an integer (default: 4)')
    parser.add_argument('--o', type=int, default=9,
                        help='the maximum number of operands in an accumulation (default: 9)')
    args = parser.parse_args()
    run(args.N, args.C, args.o)
