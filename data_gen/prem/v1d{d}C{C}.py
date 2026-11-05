'''
    Generate datasets of pseudodivision of two univariate polynomials with equal degree\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 18 Feb 2026
'''
from typing import List
from subprocess import check_output, CalledProcessError
from os import remove
from argparse import ArgumentParser
from os.path import join, abspath, dirname
import platform
import sys  # noqa
sys.path.append(join(abspath(dirname(__file__)), '..', '..'))  # noqa
from utils.folder import generate_folder  # noqa
from data_utils import PremData  # noqa
from torch import save


def run_maple_with_param(N: int, d: int, C: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        d (int): The degree of a polynomial.
        C (int): The maximum number of digit(s) in a coefficient in input polynomials.
    '''
    maple_script = f'''
LegalCheck := proc(f::polynom, x::name, deg::integer)
    if degree(f, x) = deg then
        return true:
    end if:
    return false:
end proc:

RandPoly := proc(x::name, deg::integer)
    local f:
    while true do
        f := randpoly(x, coeffs=rand(-{10 ** C - 1}..{10 ** C - 1}), dense, degree=deg):
        if LegalCheck(f, x, deg) then
            break:
        end if:
    end do:
    return f:
end proc:

Duplicated := proc(sys::list, L::list)
    local i:
    for i to nops(L) do
        if sys = L[i] then
            return true:
        end if:
    end do:
    return false:
end proc:

systems := []:
GenerateSample := proc(deg::integer, cl::integer)
    local f, g, sys, r, m, q:
    global systems:
    while true do
        f := RandPoly(x, deg):
        g := RandPoly(x, deg):
        sys := [f, g]:
        if not Duplicated(sys, systems) then
            systems := [op(systems), sys]:
            break:
        end if:
    end do:
    r := expand(prem(f, g, x, 'm', 'q')):
    m := expand(m):
    q := expand(q):
    print(f);
    print("#");
    print(g);
    print("#");
    print(cat("[", StringTools[Join](map(s -> cat("'", convert(s, string), "'"), [m, expand(m * f), q, expand(q * g), r]), ", "), "]")):
    print("#");
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
for i from 1 to {N} do
    GenerateSample({d});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)), 'v1d' +
                         str(d) + 'C' + str(C) + '.mpl')
    with open(mpl_file_path, 'w', encoding='utf-8') as file:
        file.write(maple_script)

    try:
        executor = 'cmaple' if platform.system() == 'Windows' else 'maple'
        command = f'{executor} -q {mpl_file_path}'
        output = check_output(command, shell=True, text=True)
        remove(mpl_file_path)

        muldata = output.strip().replace(' ', '').replace('\n', '').replace(
            '\\', '').replace('"', '').replace(' ', '').split('#')[:-1]
        return muldata

    except CalledProcessError as err:
        raise NotImplementedError(f'Error running Maple: {err}')


parser = ArgumentParser()
parser.add_argument('--N', type=int, default=100000,
                    help='dataset size (default: 100,000)')
parser.add_argument('--d', type=int, default=3,
                    help='the degree of a polynomial (default: 3)')
parser.add_argument('--C', type=int, default=1,
                    help='the maximum number of digit(s) in a coefficient in input polynomials (default: 1)')
args = parser.parse_args()

# Dataset folder
base_dir = join(abspath(dirname(__file__)), '..', '..', '..', 'dataset',
                'prem', 'v1d' + str(args.d) + 'C' + str(args.C))
generate_folder(base_dir)

# Generate dataset
data = run_maple_with_param(args.N, args.d, args.C)
for i in range(0, args.N * 3, 3):
    sample = PremData([data[i], data[i + 1]], eval(data[i + 2]))
    save(sample, join(base_dir, str(i // 3) + '.pt'))
